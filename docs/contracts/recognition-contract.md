# Recognition Interface Contract

This document explicitly defines the output produced by the Phase 1 (Registration) module and consumed by the Phase 2 (Recognition) module.

## REGISTRATION OUTPUT CONTRACT

**Student ID:**
`string`

**Name:**
`string`

**Individual embeddings:**
`float32` numpy array of shape `[N, 512]`

**Centroid:**
`float32` numpy array of shape `[512]`

**Normalization:**
L2 normalized

**Embedding model:**
`InceptionResnetV1` (pretrained on `vggface2`)

**Input preprocessing:**
MTCNN-based aligned face preprocessing

**Gallery loader:**
```python
from src.registration.storage import load_gallery
gallery = load_gallery()
# Returns dictionary mapping student_id to metadata and relative paths to .npy files
```

**Recognition consumer:**
Person 2 / Recognition module
