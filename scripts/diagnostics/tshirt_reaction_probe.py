"""No-ROS impact control for an ordinary fixed joint carrying cloth reaction.

Only physics:excludeFromArticulation differs between the paired lanes. Both
lanes split the same tray mass into a sliding link and its collision shell.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    sim = None
    try:
        import numpy as np
        import torch
        import omni.usd
        import omni.physics.tensors as tensors
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
        from isaaclab.sim import SimulationCfg, SimulationContext
        from src.simulation_config.scene_assets import transformed_primitives
        from src.simulation_config.isaac_cloth_object import author_cloth

        sim = SimulationContext(SimulationCfg(dt=.0025, device='cuda:0',
            gravity=(0., 0., 0.), physics_prim_path='/physicsScene', use_fabric=True))
        stage = omni.usd.get_context().get_stage()
        asset = json.loads((ROOT/'config/assets/fold_tshirt.json').read_text())
        parts = transformed_primitives(asset, [1., 1., 1.])
        material = UsdShade.Material.Define(stage, '/World/VisualMaterial')

        def cube(path, pos, size, mass, collision):
            obj = UsdGeom.Cube.Define(stage, path)
            obj.CreateSizeAttr(1.)
            obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
            obj.AddScaleOp().Set(Gf.Vec3d(*size))
            p = obj.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(p)
            UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateLinearDampingAttr(0.)
            if collision:
                UsdPhysics.CollisionAPI.Apply(p)
                c = PhysxSchema.PhysxCollisionAPI.Apply(p)
                c.CreateRestOffsetAttr(0.)
                c.CreateContactOffsetAttr(.003)
            return p

        lanes = []
        for row, excluded in enumerate((False, True)):
            for col, load in enumerate(('empty', 'rigid', 'cloth')):
                name = ('rigid_shell' if excluded else 'art_shell') + '_' + load
                path = '/World/' + name
                UsdGeom.Xform.Define(stage, path)
                x, y = col * 3., row * 3.
                anchor = cube(path+'/Anchor', (x, y, .5), (.01,)*3, 1., False)
                tray = cube(path+'/Tray', (x, y, .5), (1., 1., .04), .025, False)
                shell = cube(path+'/Shell', (x, y, .5), (1., 1., .04), .025, True)
                fixed = UsdPhysics.FixedJoint.Define(stage, path+'/RootJoint')
                fixed.CreateBody1Rel().SetTargets([anchor.GetPath()])
                fixed.CreateLocalPos0Attr(Gf.Vec3f(x, y, .5))
                UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
                art = PhysxSchema.PhysxArticulationAPI.Apply(fixed.GetPrim())
                art.CreateSolverPositionIterationCountAttr(8)
                art.CreateSolverVelocityIterationCountAttr(4)
                slide = UsdPhysics.PrismaticJoint.Define(stage, path+'/Slide')
                slide.CreateAxisAttr('Z')
                slide.CreateBody0Rel().SetTargets([anchor.GetPath()])
                slide.CreateBody1Rel().SetTargets([tray.GetPath()])
                slide.CreateLowerLimitAttr(-2.)
                slide.CreateUpperLimitAttr(2.)
                joint = UsdPhysics.FixedJoint.Define(stage, path+'/ShellJoint')
                joint.CreateBody0Rel().SetTargets([tray.GetPath()])
                joint.CreateBody1Rel().SetTargets([shell.GetPath()])
                joint.CreateExcludeFromArticulationAttr(excluded)
                entry = dict(name=name, load=load, excluded=excluded,
                             tray=str(tray.GetPath()), shell=str(shell.GetPath()))
                if load == 'cloth':
                    UsdGeom.Xform.Define(stage, path+'/Garment')
                    entry['cloth'] = author_cloth(stage, path+'/Garment', parts,
                        asset['physics'], np.array([x, y, .65, 1, 0, 0, 0]), lambda *_: material)
                elif load == 'rigid':
                    payload = cube(path+'/Payload', (x, y, .65), (.6, .6, .015), .18, True)
                    UsdPhysics.RigidBodyAPI(payload).CreateVelocityAttr(Gf.Vec3f(0, 0, -1))
                lanes.append(entry)
        stage.GetRootLayer().Export(str(out/'scene.usda'))
        sim.reset()
        view = tensors.create_simulation_view('torch')
        bodies = view.create_rigid_body_view([e[k] for e in lanes for k in ('tray', 'shell')])
        assert bodies.count == 12
        cloths = {}
        for entry in lanes:
            if entry['load'] == 'cloth':
                v = view.create_particle_cloth_view(entry['cloth'])
                vel = torch.zeros_like(v.get_velocities())
                vel.reshape(-1, 3)[:, 2] = -1.
                v.set_velocities(vel, torch.zeros(1, dtype=torch.int32, device=vel.device))
                cloths[entry['name']] = v
        rows = []
        for tick in range(241):
            if tick:
                sim.step(render=False)
            if tick % 5:
                continue
            pos = bodies.get_transforms().cpu().numpy()
            vel = bodies.get_velocities().cpu().numpy()
            row = {'tick': tick, 'lanes': {}}
            for e in lanes:
                indices = [list(bodies.prim_paths).index(e[k]) for k in ('tray', 'shell')]
                values = {'positions': pos[indices].tolist(), 'velocities': vel[indices].tolist()}
                if e['name'] in cloths:
                    points = cloths[e['name']].get_positions().cpu().numpy().reshape(-1, 3)
                    values['cloth_z'] = np.percentile(points[:, 2], [0, 50, 100]).tolist()
                row['lanes'][e['name']] = values
            rows.append(row)
        summary = {}
        for e in lanes:
            start, end = [r['lanes'][e['name']] for r in (rows[0], rows[-1])]
            summary[e['name']] = {
                'tray_dz_m': end['positions'][0][2] - start['positions'][0][2],
                'tray_vz_m_s': end['velocities'][0][2],
                'shell_tray_error_m': max(abs(r['lanes'][e['name']]['positions'][0][2] -
                    r['lanes'][e['name']]['positions'][1][2]) for r in rows)}
        passed = (abs(summary['art_shell_cloth']['tray_dz_m']) < 1e-6 and
                  summary['rigid_shell_cloth']['tray_dz_m'] < -.01 and
                  all(summary[m+'_rigid']['tray_dz_m'] < -.1 for m in ('art_shell', 'rigid_shell')) and
                  all(abs(summary[m+'_empty']['tray_dz_m']) < 1e-6 for m in ('art_shell', 'rigid_shell')))
        result = {'reaction_path_passed': passed, 'summary': summary,
                  'lanes': lanes, 'dt_s': .0025, 'trace': rows,
                  'ros_modules_loaded': [m for m in ('rclpy', 'rospy') if m in sys.modules]}
        (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        print('REACTION_PROBE', json.dumps(summary), 'PASS', passed, flush=True)
        if not passed:
            raise RuntimeError('Reaction-path impact control failed')
    except BaseException:
        import traceback
        (out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if sim is not None:
            sim._disable_app_control_on_stop_handle = True
        app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()
