# 360 visibility handling

Freeze-frame counts are visibility-conditioned. Low visible defender counts must not be interpreted as actual low defender counts unless the relevant local region is visible. Current code uses visible-area polygons plus clipped 5m/10m action buffers to decide whether local regions are fully visible. These controls still cannot infer off-camera players or convert visible counts into full-pitch availability claims.
