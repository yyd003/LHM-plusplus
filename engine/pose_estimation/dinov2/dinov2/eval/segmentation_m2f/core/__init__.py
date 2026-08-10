# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Mask2Former core compatibility exports for MMSegmentation 1.x.

MMSegmentation 1.x removed the old ``mmseg.core`` package.  Evaluation
components now live under ``mmseg.evaluation`` while common helpers live under
``mmseg.utils``.  The DINOv2-specific anchor, box and distributed utilities
remain local to avoid depending on removed legacy namespaces.
"""

from mmseg.evaluation import *  # noqa: F403
from mmseg.utils import *  # noqa: F403

from .anchor import *  # noqa: F403
from .box import *  # noqa: F403
from .utils import *  # noqa: F403
