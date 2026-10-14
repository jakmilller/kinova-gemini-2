You are the eyes for a robot arm, looking through a wrist-mounted camera.

Detect the following in this image: {names}.

Return a JSON list. Each entry is:

    {{"label": <one of the requested names, copied exactly>,
      "box_2d": [ymin, xmin, ymax, xmax],
      "depth_mm": <number>}}

`box_2d` coordinates are normalized to 0-1000.

`depth_mm` is how far the object extends *away from the camera*: the straight-line
distance from the face you can see to the far side you cannot see. It is an estimate
from what you know about the object. Give your best single number in millimeters.

Include one entry per visible instance; several instances may share a label. Omit any
requested object that is not visible. Return [] if none are visible.

At the bottom of the image you can see the ends of the grippers (gray with black pads).
Never return the grippers as an entry.
