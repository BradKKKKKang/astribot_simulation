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
