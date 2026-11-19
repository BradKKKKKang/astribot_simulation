"""Physics-invariant checks for diagnostic shells; no simulator or ROS needed."""
import json
import numpy as np
import pytest

from scripts.diagnostics.analyze_tshirt_reaction import patch_metrics


def test_shell_pair_preserves_total_inertia_and_single_active_collision(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scripts.diagnostics.tshirt_reaction import author_shells, FINGERS

    stages = []
    for mode in ('art_shell', 'rigid_shell'):
        stage = Usd.Stage.CreateInMemory()
        for i, name in enumerate(FINGERS):
            body = UsdGeom.Xform.Define(stage, '/World/' + name)
            body.AddTranslateOp().Set(Gf.Vec3d(i, 2, 3))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
            mass.CreateMassAttr(.048)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001, .002, .003))
            mass.CreateCenterOfMassAttr(Gf.Vec3f(.01, 0, .03))
            collider = UsdGeom.Cube.Define(stage, str(body.GetPath())+'/collisions')
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        directory = tmp_path / mode
        directory.mkdir()
        records = author_shells(stage, mode, directory)
        for r in records:
            original, shell = [stage.GetPrimAtPath(r[k]) for k in ('original', 'shell')]
            masses = [UsdPhysics.MassAPI(p) for p in (original, shell)]
            assert sum(m.GetMassAttr().Get() for m in masses) == pytest.approx(.048)
            np.testing.assert_allclose(sum(np.array(m.GetDiagonalInertiaAttr().Get()) for m in masses), [.001, .002, .003])
            assert masses[0].GetCenterOfMassAttr().Get() == masses[1].GetCenterOfMassAttr().Get()
            assert not UsdPhysics.CollisionAPI(stage.GetPrimAtPath(r['source_collider'])).GetCollisionEnabledAttr().Get()
            assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath(r['shell_collider'])).GetCollisionEnabledAttr().Get()
            cache = UsdGeom.XformCache()
            np.testing.assert_allclose(cache.GetLocalToWorldTransform(original), cache.GetLocalToWorldTransform(shell))
            joint = UsdPhysics.Joint(stage.GetPrimAtPath(r['joint']))
            assert joint.GetExcludeFromArticulationAttr().Get() == (mode == 'rigid_shell')
            # Normalize the intervention; all authored/composed physics must match.
            joint.GetExcludeFromArticulationAttr().Set(False)
        stages.append(stage)
    assert stages[0].Flatten().ExportToString() == stages[1].Flatten().ExportToString()


def test_fixed_material_patch_separates_lift_from_relative_slip(tmp_path):
    (tmp_path/'particles').mkdir()
    names = np.array(['astribot_arm_right_link_7'])
    start = np.array([[0., 0, .8], [.01, 0, .8]])
    wrist = np.array([[0., 0, .8, 1, 0, 0, 0]])
    np.savez(tmp_path/'particles/000930.npz', points=start, body_names=names, body_poses=wrist)
    wrist[0, 2] += .02
    # Wrist lifts 20 mm; material lifts 15 mm, hence 5 mm of downward slip.
    np.savez(tmp_path/'particles/001230.npz', points=start+[0, 0, .015], body_names=names, body_poses=wrist)
    result = patch_metrics(tmp_path, np.array([0, 1]))
    assert result['patch_lift_mm'] == pytest.approx(15.)
    assert result['patch_slip_mm'] == pytest.approx(5.)


def test_gap_feedback_uses_previous_boundary_and_limits_command_windup(tmp_path):
    from scripts.diagnostics.tshirt_reaction import GapFeedback

    def state(step, gap):
        return json.dumps({'event': 'state', 'step_id': step, 'diagnostics': {
            'cloth_proximity': {'tshirt': {'right': {'closing_axis_gap_m': gap}}}}}) + '\n'

    reference, current = tmp_path/'reference.jsonl', tmp_path/'current.jsonl'
    reference.write_text(''.join(state(step, .006) for step in range(1231)))
    current.write_text(state(850, .008) + state(851, .008) + state(852, .0055))
    controller = GapFeedback(reference, current)
    try:
        assert controller.command(851, 95.5) == pytest.approx(95.75)
        assert controller.command(852, 100.) == 100.
        assert controller.bias == 0.
        assert controller.command(853, 100.) == pytest.approx(99.9)
        with pytest.raises(RuntimeError, match='feedback missing'):
            controller.command(854, 100.)
    finally:
        controller.close()
    reference.write_text(state(0, .006))
    with pytest.raises(ValueError, match='complete'):
        GapFeedback(reference, current)
