"""Cloth schema and unbiased four-region release without simulator imports."""
import copy
import numpy as np
import pytest
from tests.unit.test_scene_assets import scene,profile
from src.simulation_config.task_reset import validate_task_reset,sample_task_objects
from src.simulation_config.scene_assets import validate_asset


def reset_cfg():
 return {'objects':['pen'],'placement':{'type':'uniform_in_regions','surface':'table.tabletop','regions':[
  {'id':f'R{i}','min':[x,y],'max':[x+.08,y+.08]} for i,(x,y) in enumerate([(-.4,-.2),(.2,-.2),(.2,.1),(-.4,.1)],1)]},'settle_steps':20}


def test_region_draw_is_seeded_and_covers_every_region(scene):
 cfg=reset_cfg(); seen=set()
 for seed in range(60):
  a=sample_task_objects(scene,cfg,np.random.default_rng(seed))['pen']
  b=sample_task_objects(scene,cfg,np.random.default_rng(seed))['pen']
  assert a==b
  expected=int(np.random.default_rng(seed).integers(4))
  assert a['reset_region']==cfg['placement']['regions'][expected]['id']
  seen.add(a['reset_region'])
 assert len(seen)==4


@pytest.mark.parametrize('field,value',[('settle_steps',-1),('settle_steps',True),('settle_steps',5001)])
def test_invalid_settling_rejected(scene,field,value):
 cfg=reset_cfg();cfg[field]=value
 with pytest.raises(ValueError):validate_task_reset(scene,cfg)


def test_duplicate_regions_rejected(scene):
 cfg=reset_cfg();cfg['placement']['regions'][1]['id']='R1'
 with pytest.raises(ValueError,match='unique'):validate_task_reset(scene,cfg)


def test_infeasible_selected_region_is_not_silently_replaced(scene):
 cfg=reset_cfg();cfg['placement']['regions']=[{'id':'outside','min':[5,5],'max':[6,6]}]
 with pytest.raises(ValueError,match='supported placement'):sample_task_objects(scene,cfg,np.random.default_rng(0))


def test_cloth_requires_explicit_valid_material(tmp_path):
 a=profile(tmp_path/'cloth.glb');a['physics']={'body_type':'cloth','collision':'particle_cloth','mass':.18,'friction':.55,'cloth':{'stretch_stiffness':10000.,'shear_stiffness':1000.,'bend_stiffness':.08,'spring_damping':.02,'velocity_damping':.005,'contact_offset':.003,'rest_offset':.0012,'solver_iterations':32}}
 validate_asset(a)
 for key,value in [('solver_iterations',0),('contact_offset',.001),('stretch_stiffness',float('nan'))]:
  bad=copy.deepcopy(a);bad['physics']['cloth'][key]=value
  with pytest.raises(ValueError):validate_asset(bad)


def test_managed_substeps_preserve_one_control_period():
 import ast
 from pathlib import Path
 from types import SimpleNamespace
 path=Path(__file__).parents[2]/'src/astribot_envs/astribot_isaaclab_env.py'
 cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='managed_step' for m in n.body))
 method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='managed_step')
 ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
 for count in [1,4]:
  ticks=[];updates=[]
  env=SimpleNamespace(managed_evaluation=True,physics_substeps=count,dt=.01,robot_joint_map={'arm':['j']},
   sim=SimpleNamespace(is_playing=lambda:True,step=lambda **kw:ticks.append(kw)),
   scene=SimpleNamespace(write_data_to_sim=lambda:None,update=lambda dt:updates.append(dt)),
   _apply_control=lambda:None,managed_publish_feedback=lambda:None,_managed_video_tick=lambda:None)
  ns['managed_step'](env,{'arm':{'position':[0.],'velocity':[0.]}})
  assert len(ticks)==count and sum(updates)==pytest.approx(.01)


def grasp_cfg():
 return {'object':'pen','point':'center','radius_m':.02,'hold_steps':250}


@pytest.mark.parametrize('change', [
 {'object':'table'}, {'object':[]}, {'point':'corner'}, {'radius_m':0},
 {'radius_m':True}, {'radius_m':float('nan')}, {'radius_m':float('inf')},
 {'hold_steps':0}, {'hold_steps':True}, {'hold_steps':5001}, {'extra':1},
])
def test_invalid_cloth_grasp_rejected(scene,change):
 scene['objects']['pen']['asset']['physics']['body_type']='cloth'
 cfg=reset_cfg();cfg['cloth_grasp']={**grasp_cfg(),**change}
 with pytest.raises(ValueError,match='cloth_grasp'):validate_task_reset(scene,cfg)


def test_cloth_grasp_requires_cloth_and_free_settling(scene):
 cfg=reset_cfg();cfg['cloth_grasp']=grasp_cfg()
 with pytest.raises(ValueError,match='cloth object'):validate_task_reset(scene,cfg)
 scene['objects']['pen']['asset']['physics']['body_type']='cloth'
 validate_task_reset(scene,cfg)
 cfg['settle_steps']=0
 with pytest.raises(ValueError,match='settling after release'):validate_task_reset(scene,cfg)


def test_center_grasp_selects_body_patch_and_rejects_whole_garment():
 from src.simulation_config.isaac_cloth_object import center_grasp_particles
 points=np.array([[-.3,-.3,0],[.3,-.3,0],[.3,.3,0],[-.3,.3,0],
                  [0,0,-.003],[0,0,.003],[.01,0,0]])
 assert center_grasp_particles(points,.02).tolist()==[4,5,6]
 assert center_grasp_particles(points+[4,2,1],.02).tolist()==[4,5,6]
 with pytest.raises(ValueError,match='free particles'):center_grasp_particles(points,1.)


def test_grasp_release_preserves_draped_shape_and_reset_removes_old_pins():
 torch=pytest.importorskip('torch')
 from src.simulation_config.isaac_cloth_object import begin_cloth_grasp,release_cloth_grasp,reset_cloth
 class View:
  def __init__(self):
   self.positions=torch.tensor([[[-.2,0,1],[0,0,1],[.2,0,1]]]).reshape(1,-1)
   self.velocities=torch.zeros_like(self.positions)
   self.masses=torch.tensor([[.1,.2,.3]])
  def get_positions(self):return self.positions
  def get_masses(self):return self.masses
  def set_positions(self,values,indices):self.positions.copy_(values)
  def set_velocities(self,values,indices):self.velocities.copy_(values)
  def set_masses(self,values,indices):self.masses.copy_(values)
 view=View();rest=np.array([[-.2,0,0],[0,0,0],[.2,0,0]])
 entry={'view':view,'particle_rest':rest,'initial_pose':[0,0,1,1,0,0,0]}
 entries={'pen':entry};original_masses=view.masses.clone()
 begin_cloth_grasp(entries,grasp_cfg())
 assert view.masses[0,1].item()==0.
 torch.testing.assert_close(view.masses[0,[0,2]],original_masses[0,[0,2]])
 view.positions[0,2]-=.15;view.velocities[0,2]=-.7
 draped=view.positions.clone();velocity=view.velocities.clone()
 release_cloth_grasp(entries);release_cloth_grasp(entries)
 torch.testing.assert_close(view.masses,original_masses)
 torch.testing.assert_close(view.positions,draped)
 torch.testing.assert_close(view.velocities,velocity)
 begin_cloth_grasp(entries,grasp_cfg())
 reset_cloth(entries,None)
 torch.testing.assert_close(view.masses,original_masses)
 assert 'grasp_masses' not in entry and 'reset_grasp' not in entry
 assert torch.count_nonzero(view.velocities)==0


@pytest.mark.parametrize('fail_tick',[None,3])
def test_initialization_releases_before_settling_and_on_error(monkeypatch,fail_tick):
 import ast
 from pathlib import Path
 from types import SimpleNamespace
 import src.simulation_config.isaac_cloth_object as cloth_module
 import src.simulation_config.isaac_scene_objects as scene_module
 import src.simulation_config.task_reset as reset_module
 path=Path(__file__).parents[2]/'src/astribot_envs/astribot_isaaclab_env.py'
 cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)
          and any(isinstance(m,ast.FunctionDef) and m.name=='_restore_initial_state' for m in n.body))
 method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_restore_initial_state')
 ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
 state={'pinned':False};ticks=[];updates=[]
 monkeypatch.setattr(cloth_module,'begin_cloth_grasp',lambda *a:state.update(pinned=True))
 monkeypatch.setattr(cloth_module,'release_cloth_grasp',lambda *a:state.update(pinned=False))
 monkeypatch.setattr(scene_module,'reset_scene_objects',lambda *a,**kw:None)
 monkeypatch.setattr(reset_module,'sample_task_objects',lambda *a:{})
 def tick(**kwargs):
  ticks.append(state['pinned'])
  if len(ticks)==fail_tick:raise RuntimeError('physics failed')
 cfg={'cloth_grasp':{**grasp_cfg(),'hold_steps':2},'settle_steps':3}
 env=SimpleNamespace(resolved_robot_placement=None,task_reset=cfg,task_scene={},np_random=None,
  scene_object_keys={'cloth':{}},physics_substeps=4,dt=.01,
  robot=SimpleNamespace(data=SimpleNamespace(default_joint_pos=0,default_joint_vel=0),
    write_joint_state_to_sim=lambda *a:None,set_joint_position_target=lambda *a:None),
  scene=SimpleNamespace(write_data_to_sim=lambda:None,update=lambda dt:updates.append(dt)),
  sim=SimpleNamespace(step=tick,forward=lambda:None,
    physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda:None)))
 if fail_tick:
  with pytest.raises(RuntimeError,match='physics failed'):ns['_restore_initial_state'](env)
 else:
  ns['_restore_initial_state'](env)
  assert ticks==[True]*8+[False]*12
 assert not state['pinned']
