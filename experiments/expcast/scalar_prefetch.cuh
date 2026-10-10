// Small scalar state only; no score/probability/PV accumulator is retained.
// Probability encoding constants and FP32 multiplication order follow vc28.
struct VCExpCastScalars {
  float dequant_scale;
  float score_scale;
  float gain;
  // A full register avoids byte extraction/packing for the carried predicate.
  uint32_t scale_safe;
};

__device__ __forceinline__ VCExpCastScalars vc_expcast_scalars(
    float q_scale, float k_scale, float attention_scale) {
  VCExpCastScalars s;
  // Preserve (Q_scale * K_scale), then attention_scale * dequant_scale.
  // Factoring attention_scale into Q_scale would change FP32 rounding.
  s.dequant_scale = q_scale * k_scale;
  s.score_scale = attention_scale * s.dequant_scale;
  s.gain = __fmul_rn(s.score_scale,11.541560173034668f);
  s.scale_safe = s.score_scale>=1.1754943508222875e-38f && isfinite(s.gain);
  return s;
}
