# Interpolation Segmentation Plan

## Goal

현재 시스템은 `MONAI/endoscopic_tool_segmentation` 기반 분할 성능은 충분히 유망하지만, 모든 비디오 프레임에 대해 직접 추론하는 방식은 처리 속도 요구사항을 만족시키기 어렵다.

이번 개선의 목표는 다음과 같다.

1. 기존 MONAI 모델을 고정밀 기준 모델로 유지한다.
2. 이전 프레임 정보와 현재 프레임 영상을 사용해 현재 프레임 마스크를 빠르게 예측하는 경량 interpolation 모델을 추가한다.
3. 실제 추론 시 MONAI 모델은 매 `N` 프레임마다 1회만 실행하고, 그 사이 프레임은 interpolation 모델이 담당하게 한다.
4. 이 하이브리드 방식이 기존 방식 대비 속도를 얼마나 개선하는지, 그리고 성능 저하가 허용 가능한지 정량적으로 평가한다.

고정 전제:

- interpolation 모델의 학습/평가 입력 해상도는 기존 추론과 동일한 `480x736`으로 유지한다.
- 사람이 수작업으로 라벨링하지 않고, 기존 MONAI 모델의 예측을 pseudo-label로 사용한다.
- 학습 데이터는 `data/video/` 아래 동영상들에서 추출한다.

## Current System Summary

현재 비디오 처리 흐름은 아래와 같다.

```text
Video frame
-> resize_with_padding(..., 480x736)
-> MonaiToolSegmenter.analyze_image(...)
-> geometry extraction
-> SimpleToolTracker.update(...)
-> overlay rendering
```

핵심 특징:

- 모든 프레임이 동일한 MONAI 세그멘터를 통과한다.
- 분할 결과는 곧바로 contour, axis, tip 계산에 사용된다.
- tracker는 분할 결과의 geometry에 의존한다.
- video pipeline은 상태를 갖는 session 기반 구조이므로, 이전 프레임 정보를 캐시하는 하이브리드 세그멘터를 넣기 좋은 구조다.

## Proposed Hybrid System

새 시스템은 두 종류의 세그멘터를 함께 사용한다.

### 1. Anchor segmenter

기존 MONAI 모델.

역할:

- 고정밀 pseudo-label 생성
- 하이브리드 추론의 기준점(anchor) 제공
- 장기 drift 누적 방지

### 2. Interpolation segmenter

새로 추가할 경량 모델.

입력:

- 이전 프레임 RGB 영상: `I_(t-1)` with shape `3x480x736`
- 이전 프레임 분할 마스크: `M_(t-1)` with shape `1x480x736`
- 현재 프레임 RGB 영상: `I_t` with shape `3x480x736`

권장 입력 처리 방식:

- raw input 관점에서는 총 `7` 채널 정보가 들어온다.
- 하지만 네트워크의 첫 층에서 이를 단일 `7-channel` 텐서로 바로 섞어 처리하지 않는다.
- 대신 아래의 3-branch encoder 구조를 사용한다.
  - previous RGB branch: `I_(t-1)` 전용 encoder
  - current RGB branch: `I_t` 전용 encoder
  - previous mask branch: `M_(t-1)` 전용 encoder

설계 원칙:

- previous RGB branch와 current RGB branch는 같은 feature extractor를 공유한다.
- 즉, 두 RGB branch는 Siamese-style shared-weight encoder를 사용한다.
- previous mask는 의미가 다른 입력이므로 1-channel 전용 encoder를 별도로 둔다.
- decoder 직전에는 3개 branch의 encoded feature를 채널 방향으로 합쳐서 fused latent representation을 만든다.
- decoder는 이 fused representation을 바탕으로 현재 프레임 마스크를 복원한다.

참고:

- 사용자 요청에서 말한 "입력 정보량이 기존보다 훨씬 많다"는 점은 타당하다.
- 다만 구조적으로는 초반부터 단순 concat 후 처리하는 방식보다, branch별 역할을 분리하는 편이 더 바람직하다.
- 구현 설명에서는 "총 7채널 정보를 입력으로 사용하되, branch별 encoder로 분리 처리 후 fusion"이라는 표현을 기준으로 한다.

출력:

- 현재 프레임의 foreground logit 또는 2-class segmentation logit
- 최종적으로는 현재 프레임 이진 분할 마스크 `M_t`

### 3. Runtime policy

하이브리드 모드에서 비디오 프레임 `t` 처리 규칙:

- `t == 0`: MONAI anchor 실행
- `t % N == 0`: MONAI anchor 실행
- 그 외: interpolation 모델 실행

state cache:

- 마지막 anchor 혹은 interpolation 결과 마스크
- 마지막 처리 프레임 RGB
- 마지막 사용 모델 종류

이 구조의 장점:

- MONAI 호출 횟수를 크게 줄일 수 있다.
- 프레임 간 변화량이 작은 구간에서 빠른 추론이 가능하다.
- anchor 프레임이 주기적으로 drift를 교정한다.

## New Model Design

## Design Intent

이 모델은 기존 MONAI보다 "더 쉬운 문제"를 푼다.

- 이전 프레임에서 도구가 어디 있었는지 이미 알고 있다.
- 현재 프레임과 이전 프레임의 시각적 차이는 상대적으로 작다.
- 따라서 거대한 encoder-decoder 대신, 얕고 가벼운 fully convolutional network로 충분할 가능성이 높다.

핵심 목표:

- 낮은 latency
- 적은 parameter 수
- GPU에서 높은 FPS
- 경미한 motion, blur, lighting change에 대한 강건성

## Recommended architecture

우선 구현 후보는 `TemporalInterpolationUNetLite` 이다.

출력:

- `1x480x736` foreground logit

권장 구조:

```text
Previous RGB branch
  Input: 3x480x736
  -> Encoder-1: lightweight conv block, stride 2
  -> Encoder-2: lightweight conv block, stride 2
  -> Encoder-3: lightweight conv block, stride 2

Current RGB branch
  Input: 3x480x736
  -> Encoder-1: same architecture as previous RGB branch
  -> Encoder-2: same architecture as previous RGB branch
  -> Encoder-3: same architecture as previous RGB branch
  -> weights are shared with previous RGB branch

Previous mask branch
  Input: 1x480x736
  -> Mask Encoder-1: lightweight conv block, stride 2
  -> Mask Encoder-2: lightweight conv block, stride 2
  -> Mask Encoder-3: lightweight conv block, stride 2

Fusion
  -> concatenate the deepest encoded features from all 3 branches along channels
  -> optional 1x1 or 3x3 fusion block to compress and mix channels

Decoder
  -> Decoder-1: upsample + skip fusion
  -> Decoder-2: upsample + skip fusion
  -> Decoder-3: upsample + skip fusion
  -> Refinement: Conv3x3
  -> Head: Conv1x1 -> 1x480x736
```

세부 원칙:

- MONAI보다 훨씬 얕은 네트워크를 사용한다.
- standard conv 대신 depthwise-separable conv를 적극 사용한다.
- spatial resolution을 너무 과도하게 줄이지 않는다.
- 다만 빠른 수술도구 이동을 더 넓은 receptive field로 다루기 위해, 기존 2-stage downsampling안보다 3-stage encoder와 3-stage decoder가 더 적절하다.
- 이전 프레임 RGB와 현재 프레임 RGB는 같은 feature extractor 관점에서 처리한다.
- 따라서 두 RGB branch는 가중치를 공유한다.
- 이전 마스크는 성격이 다르므로 별도의 1-channel branch에서 처리한다.
- decoder에 들어가기 직전, 3개 branch의 deepest feature를 채널 방향으로 합친다.
- decoder는 이 fused latent representation을 기준으로 현재 프레임 마스크를 복원한다.
- 필요하다면 decoder 각 단계에서 branch별 skip feature도 함께 fusion할 수 있지만, 첫 구현은 deepest fusion 중심의 단순 구조로 시작하는 것이 안전하다.

예상 장점:

- 빠른 motion에 대해 더 넓은 receptive field를 확보할 수 있다.
- 시점별 RGB 정보를 초반부터 섞지 않으므로 temporal role 분리가 명확하다.
- shared-weight RGB encoder 덕분에 parameter 수 증가를 억제할 수 있다.
- previous mask branch가 coarse prior를 별도로 안정적으로 표현할 수 있다.

## Why this architecture first

이유는 다음과 같다.

1. 빠른 도구 이동이 있을 때 2-level encoder보다 3-level encoder가 더 넓은 receptive field를 제공한다.
2. 이전 프레임 RGB와 현재 프레임 RGB를 초반부터 섞지 않고 분리 인코딩하면, 각 branch가 시점별 appearance를 더 안정적으로 표현할 수 있다.
3. 두 RGB branch의 가중치를 공유하면 parameter 수를 크게 늘리지 않으면서 Siamese-style temporal comparison 효과를 얻을 수 있다.
4. 이전 마스크는 의미적으로 영상과 다르므로 별도 branch가 더 자연스럽다.
5. deepest feature fusion 후 공통 decoder를 쓰면 구조가 비교적 단순하고, 이후 skip-level fusion 추가 같은 확장도 가능하다.

## Training Strategy

## Supervision source

학습 정답은 수작업 라벨이 아니라 MONAI 예측을 사용한다.

프레임 쌍 구성:

- 입력:
  - `I_(t-1)`
  - `PseudoMask_(t-1)` from MONAI
  - `I_t`
- 타깃:
  - `PseudoMask_t` from MONAI

즉, 학습 목표는 아래와 같다.

```text
f(I_(t-1), PseudoMask_(t-1), I_t) -> PseudoMask_t
```

## Dataset generation plan

`data/video/*.mp4`에서 학습 샘플을 만든다.

처리 절차:

1. 비디오를 순차적으로 읽는다.
2. 각 프레임을 현재 시스템과 동일하게 `480x736`으로 resize/pad 한다.
3. MONAI로 각 프레임의 pseudo-mask를 생성한다.
4. `(prev_rgb, prev_mask, curr_rgb, curr_mask)` 샘플을 저장하거나 on-the-fly로 만든다.
5. 영상 단위 또는 프레임 범위 단위로 train, val, test split을 고정한다.

권장 split 전략:

- train: 대부분의 비디오
- val: 소수의 비디오
- test: 완전히 분리된 비디오

중요 원칙:

- 같은 비디오의 인접 프레임이 train과 test에 동시에 섞이지 않도록 한다.
- split은 frame 단위가 아니라 video 단위로 나누는 것이 안전하다.

## Data storage strategy

이번 프로젝트에서는 pseudo-label dataset을 별도 파일로 생성하지 않는다.

결정 이유:

- 원본 비디오 용량이 매우 크다.
- 모든 학습 프레임 또는 pseudo-label을 별도 파일로 저장하면 저장 공간이 빠르게 부족해질 수 있다.
- augmentation도 학습 시점 메모리에서 바로 수행하는 편이 더 단순하다.

채택 전략:

- 데이터셋은 `data/video/*.mp4`에서 바로 이전 프레임과 현재 프레임을 읽는다.
- train split에서는 synchronized augmentation을 메모리에서 즉시 적용한다.
- MONAI teacher는 학습 loop 안에서 현재 batch에 대해 pseudo-label을 즉시 생성한다.
- 즉, pseudo-label은 장기 저장 파일이 아니라 training step 내부 intermediate로만 존재한다.

tradeoff:

- 장점:
  - 저장 공간 사용량을 크게 줄일 수 있다.
  - augmentation과 pseudo-label 생성을 항상 현재 설정에 맞춰 적용할 수 있다.
- 단점:
  - teacher MONAI 추론 비용이 학습 시간에 직접 포함된다.
  - cached dataset 대비 epoch 시간이 더 길 수 있다.

현재 우선순위에서는 저장 공간 절약과 구현 단순성이 더 중요하므로, on-the-fly 방식이 기본 전략이다.

## Loss functions

초기 loss는 단순하고 안정적인 조합으로 시작한다.

- `BCEWithLogitsLoss`
- soft Dice loss

최종 loss 예시:

```text
loss = 0.5 * BCE + 0.5 * Dice
```

추가 후보:

- anchor frame과 멀어질수록 confidence decay를 반영한 weighting
- boundary-aware loss

하지만 첫 단계에서는 넣지 않는다.

## Augmentation

입력은 시간적으로 연관된 쌍이므로, augmentation은 frame pair와 mask에 동기화되어야 한다.

초기 augmentation 후보:

- horizontal flip
- brightness, contrast jitter
- small rotation
- mild gaussian blur

주의:

- 이전 프레임과 현재 프레임에 서로 다른 기하학 augmentation을 적용하면 temporal consistency가 깨지므로 피한다.

## Evaluation Plan

## Main questions

실험은 아래 질문에 답해야 한다.

1. interpolation 모델 단독 추론 속도는 충분히 빠른가?
2. 하이브리드 모드에서 전체 FPS는 얼마나 개선되는가?
3. MONAI-only 대비 segmentation quality는 얼마나 유지되는가?
4. anchor interval `N`이 커질수록 drift가 얼마나 증가하는가?

## Baselines and variants

비교 대상:

1. `MONAI-only`
   - 모든 프레임에서 MONAI 실행
2. `Hybrid-N`
   - 매 `N` 프레임마다 MONAI 실행
   - 그 사이는 interpolation 모델 실행

우선 비교할 `N` 후보:

- `2`
- `3`
- `5`
- `10`

이 값들은 너무 촘촘하지 않으면서도 speed-quality tradeoff를 확인하기에 적절하다.

## Segmentation metrics

정량 비교에는 아래 지표를 사용한다.

- IoU
- Dice
- pixel precision
- pixel recall

실험 1차 기준 정답:

- 수작업 GT가 없으므로, test split에서도 MONAI 예측을 비교 기준으로 사용할 수밖에 없다.

이 한계는 분명히 문서화해야 한다.

의미:

- 이 실험은 "MONAI teacher를 얼마나 잘 근사하는가"를 측정한다.
- 절대적 실제 정확도가 아니라, teacher consistency와 runtime tradeoff를 보는 실험이다.

## Runtime metrics

반드시 함께 기록할 항목:

- average FPS
- median per-frame latency
- p95 per-frame latency
- MONAI invocation count
- interpolation invocation count

가능하면 추가:

- GPU memory usage

## Failure analysis

수치만으로 부족하므로 qualitative review도 필요하다.

검토 대상:

- 급격한 camera motion
- motion blur
- occlusion
- smoke, blood, glare
- tool이 frame 밖으로 나갔다가 다시 들어오는 상황
- 여러 도구가 겹치는 상황

결과물:

- representative failure case image 혹은 short clip export

## Integration Plan

## New components to add

권장 추가 디렉터리와 모듈:

```text
app/
  services/
    segmentation/
      monai_segmenter.py
      interpolation_segmenter.py
      hybrid_segmenter.py
  training/
    datasets/
      temporal_pseudolabel_dataset.py
    models/
      temporal_interpolation_unet_lite.py
    losses/
      segmentation_losses.py
    scripts/
      build_pseudolabel_dataset.py
      train_interpolation_model.py
      evaluate_segmentation_models.py
```

## Segmentation abstraction

현재는 `VideoPipelineSession`이 `MonaiToolSegmenter`에 직접 의존한다.
새 구조에서는 공통 인터페이스 또는 duck-typed contract가 필요하다.

필수 capability:

- `segment_mask(image_rgb) -> mask`
- 또는
- `analyze_image(...) -> FrameResult`

하지만 interpolation 모델은 이전 프레임 상태가 필요하므로, video용 API는 더 명시적으로 바꾸는 편이 낫다.

권장 방향:

- still image는 기존 `MonaiToolSegmenter` 유지
- video는 `HybridVideoSegmenter` 같은 stateful 객체를 도입

예상 video API:

```python
processed_result = video_segmenter.analyze_video_frame(
    image_rgb=current_frame,
    original_image_size=original_size,
    mapping=(scale, pad),
    frame_index=t,
)
```

내부에서:

- mode가 `monai_only`면 MONAI 실행
- mode가 `hybrid`면 anchor interval 정책에 따라 MONAI 또는 interpolation 실행
- 이전 frame RGB, mask cache 업데이트

## GUI changes

GUI와 설정에서 아래 항목을 조절 가능해야 한다.

- segmentation mode
  - `MONAI only`
  - `Hybrid`
- hybrid anchor interval
  - 예: `2`, `3`, `5`, `10`

표시 정보 후보:

- current frame에서 어떤 모델이 사용되었는지
- current hybrid interval
- processing FPS

## Experimental Deliverables

최소 산출물:

1. pseudo-label dataset builder
2. interpolation model training script
3. evaluation script
4. 비교 결과 요약 파일
5. GUI 또는 pipeline 통합 코드

권장 추가 산출물:

- 실험 결과 CSV
- representative visualization images
- best checkpoint 저장 규칙

## Risks and Mitigations

## Risk 1. Teacher noise

MONAI 예측 자체가 완벽한 GT가 아니므로 student가 teacher의 오류까지 학습할 수 있다.

대응:

- 초기에 목표를 "teacher approximation"으로 명확히 정의한다.
- failure case를 별도 시각화로 검토한다.

## Risk 2. Drift accumulation

anchor interval이 커질수록 interpolation 오류가 누적될 수 있다.

대응:

- 여러 `N` 값을 비교한다.
- `Hybrid-10`이 너무 불안정하면 `Hybrid-5` 또는 `Hybrid-3`가 현실적 기준이 된다.

## Risk 3. Training data scale

`data/video/`는 용량이 매우 크다. pseudo-label 생성부터 시간이 오래 걸릴 수 있다.

대응:

- 먼저 소수 비디오 subset으로 pipeline을 검증한다.
- 그 다음 full dataset으로 확장한다.

## Risk 4. Runtime integration complexity

stateful interpolation은 seek, pause, replay 같은 UI 기능과 충돌할 수 있다.

대응:

- seek 시에는 cache reset 후 해당 프레임을 anchor 방식으로 다시 시작한다.
- video session이 frame index와 segmenter state를 함께 관리하도록 설계한다.

## Risk 5. Metric bias

teacher 기반 pseudo-label 평가만으로는 실제 임상적 품질을 확정할 수 없다.

대응:

- 문서에 한계를 명시한다.
- 가능하면 후속 단계에서 소규모 수작업 GT 검증을 고려한다.

## Implementation Sequence

작업 순서는 아래처럼 가져가는 것이 가장 안전하다.

### Phase 1. Planning

- 요구사항과 하이브리드 정책 확정
- model I/O와 split 전략 명세
- 실험 항목 정의

### Phase 2. Data pipeline

- video split 정의
- pseudo-label dataset builder 구현
- sample inspection 도구 구현

### Phase 3. Model and training

- 경량 interpolation 모델 구현
- loss 구현
- training loop 구현
- checkpoint, logging, validation 구현

### Phase 4. Offline evaluation

- MONAI-only baseline 측정
- Hybrid-N variants 측정
- metric, runtime 결과 수집

### Phase 5. Online integration

- hybrid video segmenter 구현
- video pipeline 수정
- GUI mode, interval control 추가

### Phase 6. Review

- qualitative failure review
- best interval recommendation
- next iteration 설계

## Acceptance Criteria

이번 큰 작업이 완료되었다고 보기 위한 기준:

1. `PLAN.md`에 설계와 실험 기준이 정리되어 있다.
2. interpolation 모델 학습 코드가 재현 가능하게 실행된다.
3. pseudo-label dataset 생성이 가능하다.
4. `MONAI-only`와 `Hybrid-N` 비교 결과가 남는다.
5. GUI 또는 video pipeline에서 `MONAI only`와 `Hybrid`를 선택할 수 있다.
6. hybrid 모드에서 anchor interval을 조절할 수 있다.

## TODO

- [ ] `PLAN.md` 내용을 기준으로 비디오 split 전략과 산출 파일 구조를 구체화한다.
- [ ] pseudo-label dataset builder의 디렉터리 구조와 저장 포맷을 설계한다.
- [ ] `TemporalInterpolationUNetLite` 경량 모델을 구현한다.
- [ ] interpolation 모델 학습용 dataset, dataloader 모듈을 구현한다.
- [ ] BCE + Dice 기반 self-supervised loss 모듈을 구현한다.
- [ ] interpolation 모델 학습 스크립트를 구현한다.
- [ ] 소규모 subset으로 pseudo-label 생성 및 학습 파이프라인 smoke test를 수행한다.
- [ ] 전체 `data/video` 기준 pseudo-label dataset 생성 파이프라인을 실행 가능하게 정리한다.
- [ ] `MONAI-only`, `Hybrid-2`, `Hybrid-3`, `Hybrid-5`, `Hybrid-10` 오프라인 평가 스크립트를 구현한다.
- [ ] segmentation 성능 지표와 runtime 지표를 CSV 또는 markdown 결과로 저장하도록 한다.
- [ ] hybrid video segmenter를 구현한다.
- [ ] video pipeline을 수정해 `MONAI only`와 `Hybrid` 모드를 지원하게 한다.
- [ ] GUI에서 segmentation mode와 anchor interval을 선택 가능하게 한다.
- [ ] seek, replay, session reset 시 hybrid cache가 안전하게 초기화되도록 한다.
- [ ] 실패 사례를 시각적으로 검토할 수 있는 export 또는 요약 기능을 추가한다.
