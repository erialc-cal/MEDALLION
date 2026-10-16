# backward-compatibility shim — new code should import from medal.teacher and medal.io
from medal.teacher import get_teacher_embeddings  # noqa: F401
from medal.io import compute_losses, eval_student  # noqa: F401
