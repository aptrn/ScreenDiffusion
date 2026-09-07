"""What the device compositor is when there is no device. Issue #31, GPU-free tier.

The Gate's third item: moving the blend onto the GPU must not move the compositor's
rules out of the merge gate. So `DeviceCompositor` *is* a `Compositor` - the actions,
the feather, the inwards ramp, the alpha cache and the numpy blend are inherited
rather than reimplemented - and every rule `tests/test_compositor.py` holds is
therefore a rule about the shipped object too, asserted here without a CUDA device.

The arithmetic half needs one, and lives in `tests/test_gpu_device_compositor.py`,
where the two implementations are held to each other byte for byte.

No torch: `device_compositor` imports it inside the two functions that touch a
device, which is what lets this module import it at all - and is asserted here,
because the moment it moves to the top the whole module leaves the merge gate.
"""

import ast
from pathlib import Path

import numpy as np

import device_compositor
from compositor import MASKED, PASSTHROUGH, Compositor, composite, feather_alpha
from device_compositor import DEVICE, HOST, DeviceCompositor
from detection import Box, Track, Tracks
from region_scheduler import RegionScheduler
from render_plan import validate_plan

W, H = 64, 48
REGION = Box(10, 8, 40, 40)

SOURCE = Path(device_compositor.__file__)


def plan_of(concept="person"):
    result = validate_plan({"source_prompt": "wet denim",
                            "targets": [{"id": "t0", "concept": concept,
                                         "region": "full_box", "box_scale": 1.0}]})
    assert result.plan is not None, result.reason
    return result.plan


def selection_of(*boxes):
    tracks = Tracks(tracks=tuple(
        Track(track_id=index, box=Box(*box), concept="person", confidence=0.9)
        for index, box in enumerate(boxes)), ticks=1)
    return RegionScheduler().select(tracks, plan_of(), W, H)


# --- the tier split ---------------------------------------------------------


def test_torch_is_imported_inside_the_functions_that_use_it():
    """It is on the frame loop's import path and the GUI process's; a torch import
    at module scope would put a CUDA runtime in a process that never touches one,
    and would take the whole module out of the merge gate's tier."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    top_level = [node for node in tree.body if isinstance(node, (ast.Import,
                                                                 ast.ImportFrom))]
    names = {alias.name for node in top_level if isinstance(node, ast.Import)
             for alias in node.names}
    names |= {node.module for node in top_level if isinstance(node, ast.ImportFrom)}
    assert "torch" not in names
    assert any(isinstance(node, ast.Import) and any(alias.name == "torch"
                                                    for alias in node.names)
               for node in ast.walk(tree)), "nothing imports torch at all"


# --- the same compositor ----------------------------------------------------


def test_the_device_compositor_is_a_compositor():
    """The reference implementation is not a second implementation: everything the
    GPU-free tier holds `Compositor` to is held about this object as well."""
    assert issubclass(DeviceCompositor, Compositor)
    assert isinstance(DeviceCompositor(), Compositor)


def test_it_decides_what_a_frame_is_exactly_as_the_host_compositor_does():
    selection = selection_of(REGION)
    host, device = Compositor().frame(selection, W, H), DeviceCompositor().frame(
        selection, W, H)
    assert device.action == host.action == MASKED
    assert np.array_equal(device.alpha, host.alpha)


def test_a_selective_plan_with_nothing_detected_still_costs_no_diffusion_call():
    assert DeviceCompositor().frame(selection_of(), W, H).action == PASSTHROUGH


def test_the_alpha_is_still_zero_outside_the_region():
    """The criterion, on the map the device path uploads."""
    alpha = DeviceCompositor().frame(selection_of(REGION), W, H).alpha
    outside = np.ones((H, W), dtype=bool)
    for box in selection_of(REGION).boxes:
        outside[box.y0:box.y1, box.x0:box.x1] = False
    assert np.all(alpha[outside] == 0.0)


def test_the_numpy_blend_is_still_reachable_on_it():
    """The host path is not deleted (the issue's third trap): the bench runners and
    anything without a device still have it, and it is what the device path is
    measured against."""
    generator = np.random.default_rng(11)
    source = generator.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    rendered = generator.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    alpha = feather_alpha([REGION], W, H)
    assert np.array_equal(DeviceCompositor().blend(source, rendered, alpha),
                          composite(source, rendered, alpha))


def test_the_two_places_a_blend_can_run_are_named_once():
    assert (HOST, DEVICE) == ("host", "device")
