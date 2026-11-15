"""LIBERO environment wrapper that reproduces the OpenVLA-OFT evaluation protocol exactly, without its side
effects, and adds the measurements this study needs.

Matches upstream (experiments/robot/libero/run_libero_eval.py @ e4287e9):
  * env created with resolution 256 and env.seed(0) (upstream notes the seed affects object poses)
  * 10 no-op "settle" steps after setting the initial state, then max_steps policy steps
  * camera frames rotated 180 degrees ([::-1, ::-1])
  * proprio = [eef_pos(3), axis-angle(eef_quat)(3), gripper_qpos(2)]
  * policy gripper output in [0,1] -> sign(2x-1) -> inverted before env.step
Differs from upstream, on purpose:
  * never mutates observation arrays (upstream quat2axisangle clips in place)
  * exceptions propagate (upstream counts any crash as a task failure)
  * optional instance segmentation for the background / target masks
  * goal predicates of *every* task in the suite are evaluated each step (intent oracle); valid because
    all LIBERO-Goal tasks share one scene (checked at construction, not assumed)
  * task-relevant entities come from the BDDL objects of interest (no hand-written lists)
  * fixture layout made order-independent: LIBERO re-samples the poses of fixed bodies (cabinet, stove, rack)
    on every reset, and they live in the MuJoCo model, not in the state, so upstream's episode k sees the
    layout of reset k+1 of a seed-0 env. reset(k) restores exactly that layout from
    assets/fixture_layouts_<suite>.npz (scripts/setup/build_fixture_layouts.py; bit-exactness tested)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

TASK_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SETTLE_STEPS = 10
DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
# Task-relevant entities are derived from the BDDL files, never hand-listed. A support surface (a BDDL fixture whose
# category is a table) is the arena's table and never a task-relevant entity.
SUPPORT_CATEGORY_SUFFIX = "table"


def support_surfaces(parsed: dict) -> set[str]:
    return {n for cat, names in parsed["fixtures"].items() if cat.endswith(SUPPORT_CATEGORY_SUFFIX) for n in names}


def entity_of(name: str, parsed: dict) -> str:
    """Map a BDDL object-of-interest (object, fixture or region) to the scene entity it belongs to.

    Regions are keyed '<target>_<region>' by LIBERO's parser and carry their target entity.
    """
    if name in parsed["regions"]:
        return parsed["regions"][name]["target"]
    entities = {n for group in (parsed["objects"], parsed["fixtures"]) for names in group.values() for n in names}
    if name not in entities:
        raise KeyError(f"{name!r} is neither an object, a fixture nor a region of this problem")
    return name


def task_entities(parsed: dict) -> set[str]:
    """Entities named by a task's objects of interest (definitional), minus support surfaces."""
    return {entity_of(n, parsed) for n in parsed["obj_of_interest"]} - support_surfaces(parsed)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Copy of robosuite's conversion (as used upstream) that does not modify its argument."""
    q = np.array(quat, dtype=np.float64, copy=True)
    q[3] = min(max(q[3], -1.0), 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def load_init_states(suite, task_id: int) -> np.ndarray:
    """The benchmark initial states of a task: LIBERO's get_task_init_states (same file), loaded with
    weights_only=False explicitly. torch < 2.6 (OFT env) used that default; torch >= 2.6 (LeRobot env) defaults to
    weights_only=True, which cannot read these pickled arrays. Same bytes, same array in both environments."""
    import torch
    from libero.libero import get_libero_path

    task = suite.tasks[task_id]
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return torch.load(path, weights_only=False)  # nosec: benchmark file from the pinned LIBERO checkout


def process_policy_action(action: np.ndarray) -> np.ndarray:
    """Upstream process_action for model_family='openvla' (binarize + invert gripper). Returns a copy."""
    a = np.array(action, dtype=np.float64, copy=True)
    a[..., -1] = np.sign(2.0 * a[..., -1] - 1.0)
    a[..., -1] *= -1.0
    return a


@dataclass(frozen=True)
class Frame:
    """One observation as the policy sees it (before policy preprocessing). Arrays are read-only."""

    agentview: np.ndarray  # uint8 256x256x3, rotated 180 deg
    wrist: np.ndarray
    proprio: np.ndarray  # float64 (8,)
    agentview_seg: Optional[np.ndarray] = None  # int instance ids, same orientation as images
    wrist_seg: Optional[np.ndarray] = None
    eef_quat: Optional[np.ndarray] = None  # robosuite xyzw; pi0.5 computes its state from it (vlamech.models.pi05)


def _ro(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is None:
        return None
    a = np.ascontiguousarray(a).copy()
    a.setflags(write=False)
    return a


class LiberoTaskEnv:
    def __init__(self, suite_name: str, task_id: int, resolution: int = 256, segmentation: bool = False,
                 settle_steps: int = SETTLE_STEPS, env_seed: int = 0, render: bool = True):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv

        self.suite_name = suite_name
        self.task_id = task_id
        self.suite = benchmark.get_benchmark_dict()[suite_name]()
        self.task = self.suite.get_task(task_id)
        self.instruction = self.task.language
        self.max_steps = TASK_MAX_STEPS[suite_name]
        bddl = os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file)
        self.bddl_file = bddl
        kwargs = dict(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
        self.segmentation = segmentation
        self.render = render
        if not render:  # state-only measurements (no frames): physics is identical, no GPU render context
            if segmentation:
                raise ValueError("segmentation needs rendering")
            from libero.libero.envs.env_wrapper import ControlEnv

            self._env = ControlEnv(**kwargs, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        elif segmentation:
            self._env = SegmentationRenderEnv(**kwargs, camera_segmentations="instance")
        else:
            self._env = OffScreenRenderEnv(**kwargs)
        self.settle_steps, self.env_seed = int(settle_steps), int(env_seed)
        self._env.seed(self.env_seed)
        # loaded once (LIBERO re-checks every configured path, printing warnings, on each load); read-only so
        # nothing downstream can modify the benchmark states in place
        self._init_states = np.array(load_init_states(self.suite, task_id), copy=True)
        self._init_states.setflags(write=False)
        self._layout = self._load_layouts()
        self._parsed = self._parse_suite()
        self._goal_states = {i: p["goal_state"] for i, p in self._parsed.items()}
        own = self._parsed[task_id]
        self.movable_objects = sorted(n for names in own["objects"].values() for n in names)
        # Union over all suite tasks: a "background" perturbation must not hide any task's candidate target,
        # otherwise measured intent substitution would depend on which objects happened to stay visible.
        self.protected_entities = sorted(set().union(*(task_entities(p) for p in self._parsed.values())))
        self.own_entities = sorted(task_entities(own))
        self._last_obs = None

    # ------------------------------------------------------------------ setup
    def _load_layouts(self) -> dict:
        path = Path(__file__).resolve().parents[3] / "assets" / f"fixture_layouts_{self.suite_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run scripts/setup/build_fixture_layouts.py")
        z = np.load(path)
        m = self.inner.sim.model
        names = [str(n) for n in z["body_names"]]
        self.layout_file = path
        return {"ids": np.array([m.body_name2id(n) for n in names], dtype=int), "names": names,
                "pos": z["body_pos"][self.task_id], "quat": z["body_quat"][self.task_id]}

    def _apply_layout(self, init_state_idx: int) -> None:
        m = self.inner.sim.model
        ids = self._layout["ids"]
        m.body_pos[ids] = self._layout["pos"][init_state_idx]
        m.body_quat[ids] = self._layout["quat"][init_state_idx]

    def _parse_suite(self) -> dict[int, dict]:
        """Parse the BDDL of every suite task that shares this task's scene (objects, fixtures, regions), so their
        goal predicates can be evaluated in this env (intent oracle). LIBERO-Goal: all 10 tasks (checked in tests)."""
        from libero.libero import get_libero_path
        import libero.libero.envs.bddl_utils as BDDLUtils

        out = {}
        for i in range(self.suite.n_tasks):
            t = self.suite.get_task(i)
            out[i] = BDDLUtils.robosuite_parse_problem(os.path.join(get_libero_path("bddl_files"), t.problem_folder,
                                                                    t.bddl_file))
        scene = lambda p: (p["objects"], p["fixtures"], sorted(p["regions"]))
        return {i: p for i, p in out.items() if scene(p) == scene(out[self.task_id])}

    @property
    def inner(self):
        """The robosuite problem env (for predicates, sim access)."""
        return self._env.env

    def initial_states(self) -> np.ndarray:
        return self._init_states

    def reset(self, init_state_idx: int) -> Frame:
        """Reset to a benchmark initial state, run the settle steps, return the first policy frame."""
        self._env.reset()
        # Upstream's fixture layout for this episode is written into the model BETWEEN reset() and
        # set_init_state(), so the call sequence afterwards is exactly upstream's (set_init_state -> settle steps).
        # Re-rendering via an extra regenerate_obs_from_state() would shift robosuite's observable sampling phase
        # and change the camera frames returned by later steps (found by the bit-exactness test).
        self._apply_layout(init_state_idx)
        obs = self._env.set_init_state(self.initial_states()[init_state_idx])
        for _ in range(self.settle_steps):
            obs, _, _, _ = self._env.step(DUMMY_ACTION.tolist())
        self._last_obs = obs
        return self._frame(obs)

    # ------------------------------------------------------------------ stepping
    def step(self, policy_action: np.ndarray, raw: bool = False) -> tuple[Frame, bool]:
        """Execute one *policy-space* action (OFT: gripper in [0,1], binarized and inverted here); returns (frame,
        instructed-task success). raw=True: the action is already in robosuite's space (pi0.5, LeRobot's eval passes
        it to env.step unchanged) and is executed as is."""
        env_action = np.array(policy_action, dtype=np.float64, copy=True) if raw else process_policy_action(policy_action)
        obs, _, done, _ = self._env.step(env_action.tolist())
        self._last_obs = obs
        return self._frame(obs), bool(done)

    def _frame(self, obs: dict) -> Optional[Frame]:
        if not self.render:
            return None
        proprio = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
        seg_a = seg_w = None
        if self.segmentation:
            seg_a = obs["agentview_segmentation_instance"][::-1, ::-1, 0]
            seg_w = obs["robot0_eye_in_hand_segmentation_instance"][::-1, ::-1, 0]
        return Frame(
            agentview=_ro(obs["agentview_image"][::-1, ::-1]),
            wrist=_ro(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
            proprio=_ro(proprio),
            agentview_seg=_ro(seg_a),
            wrist_seg=_ro(seg_w),
            eef_quat=_ro(obs["robot0_eef_quat"]),
        )

    # ------------------------------------------------------------------ state access
    def get_sim_state(self) -> np.ndarray:
        return np.array(self._env.get_sim_state(), copy=True)

    def set_sim_state(self, state: np.ndarray) -> Frame:
        """Restore a saved MuJoCo state and regenerate observations (for open-loop replay)."""
        obs = self._env.regenerate_obs_from_state(state)
        self._last_obs = obs
        return self._frame(obs)

    # ------------------------------------------------------------------ oracle
    def goal_status(self) -> dict[int, bool]:
        """Evaluate every suite task's goal predicate in the current state."""
        env = self.inner
        res = {}
        for i, goal in self._goal_states.items():
            ok = True
            for pred in goal:
                ok = bool(env._eval_predicate(pred)) and ok
            res[i] = ok
        return res

    def object_positions(self) -> dict[str, np.ndarray]:
        env = self.inner
        out = {}
        for name in self.movable_objects:  # body-frame origin (LIBERO ObjectState.get_geom_state)
            out[name] = np.array(env.object_states_dict[name].get_geom_state()["pos"], dtype=np.float64, copy=True)
        return out

    def table_top(self) -> dict:
        """Table-top plane and footprint from the COMPILED MuJoCo model: among the collision boxes of the support
        surface's bodies, the one whose upper face is highest (LIBERO-Goal: the single box 'table_collision').
        Returns {"z": top height, "xy_min": (2,), "xy_max": (2,)} in world frame (axis-aligned bounds)."""
        if not hasattr(self, "_table_top"):
            sim = self.inner.sim
            m = sim.model
            bodies = self._support_bodies()
            best = None
            for g in range(m.ngeom):
                if int(m.geom_type[g]) != 6 or not (m.geom_contype[g] or m.geom_conaffinity[g]):  # 6 = mjGEOM_BOX
                    continue
                if not set(_body_chain(m, int(m.geom_bodyid[g]))) & bodies:
                    continue
                R = np.asarray(sim.data.geom_xmat[g]).reshape(3, 3)
                half = np.abs(R) @ np.asarray(m.geom_size[g], float)  # world-frame half extents
                c = np.asarray(sim.data.geom_xpos[g], float)
                if best is None or c[2] + half[2] > best["z"]:
                    best = {"z": float(c[2] + half[2]), "xy_min": c[:2] - half[:2], "xy_max": c[:2] + half[:2]}
            if best is None:
                raise RuntimeError("no collision box found for the support surface")
            self._table_top = best
        return self._table_top

    def _support_bodies(self) -> set[int]:
        """Body ids of the arena table(s): the body named like the BDDL support surface, else the arena's 'table'."""
        m = self.inner.sim.model
        names = {m.body_id2name(b) for b in range(m.nbody)}
        out = set()
        for s in support_surfaces(self._parsed[self.task_id]):
            body = s if s in names else "table"
            if body not in names:
                raise RuntimeError(f"no body for support surface {s!r}")
            out.add(m.body_name2id(body))
        return out

    def off_table(self, pos: np.ndarray) -> bool:
        """Definitional: the object is below the table-top plane AND outside the table footprint
        (it can only be there by having fallen off). No margins: an object resting on the table is inside
        the footprint; one held in the air beyond the edge is above the plane."""
        t = self.table_top()
        outside = bool(np.any(pos[:2] < t["xy_min"]) or np.any(pos[:2] > t["xy_max"]))
        return outside and bool(pos[2] < t["z"])

    def robot_contacts(self) -> dict[str, float]:
        """Max contact normal force (N) between any robot geom and each scene entity (every BDDL object and
        fixture of the loaded scene, incl. the table) in the current state; "other" = walls, floor and anything
        outside the entities. 0.0 = not touching."""
        import mujoco

        sim = self.inner.sim
        model, data = sim.model._model, sim.data._data
        robot, owner, _ = self._geom_owners()
        out = {e: 0.0 for e in self.contact_entities} | {"other": 0.0}
        f = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            r1, r2 = c.geom1 in robot, c.geom2 in robot
            if r1 == r2:
                continue
            key = owner.get(c.geom2 if r1 else c.geom1, "other")
            mujoco.mj_contactForce(model, data, i, f)  # f[0] = normal component in the contact frame
            out[key] = max(out[key], abs(float(f[0])))
        return out

    def robot_contact_forces(self, contacts: Optional[dict] = None) -> dict[str, float]:
        """Safety-proxy summary of robot_contacts(): "object" = max over movable objects (includes normal
        grasping, so only meaningful relative to clean episodes), "environment" = max over everything else
        (table, fixtures, walls)."""
        c = self.robot_contacts() if contacts is None else contacts
        movable = set(self.movable_objects)
        return {"object": max(v for k, v in c.items() if k in movable),
                "environment": max(v for k, v in c.items() if k not in movable)}

    @property
    def contact_entities(self) -> list[str]:
        return self._geom_owners()[2]

    def _geom_owners(self) -> tuple[set, dict, list]:
        """(robot geom ids, {geom id: entity name}, entity names), from the kinematic tree of the compiled model:
        a geom belongs to the entity whose root body is its nearest ancestor."""
        if not hasattr(self, "_owners"):
            import libero.libero.envs.bddl_utils as BDDLUtils

            m = self.inner.sim.model
            p = BDDLUtils.robosuite_parse_problem(self.bddl_file)
            entities = sorted(n for group in (p["objects"], p["fixtures"]) for names in group.values() for n in names)
            body_names = [m.body_id2name(b) or "" for b in range(m.nbody)]
            robot_roots = {b for b in range(m.nbody) if body_names[b].startswith(("robot0_", "gripper0_", "mount0_"))}
            root = {}
            support = support_surfaces(p)
            names = set(body_names)
            for e in entities:
                if e in support:  # the BDDL table is the arena's table body, not a LIBERO object
                    root[m.body_name2id(e if e in names else "table")] = e
                    continue
                obj = self.inner.get_object(e)
                if obj is None:
                    raise RuntimeError(f"BDDL entity {e!r} has no compiled object")
                root[m.body_name2id(obj.root_body)] = e
            robot, owner = set(), {}
            for g in range(m.ngeom):
                for b in _body_chain(m, int(m.geom_bodyid[g])):  # nearest ancestor first
                    if b in robot_roots:
                        robot.add(g)
                        break
                    if b in root:
                        owner[g] = root[b]
                        break
            if not robot or not set(self.movable_objects) <= set(owner.values()):
                raise RuntimeError("could not identify robot / object geoms")
            self._owners = (robot, owner, entities)
        return self._owners

    # ------------------------------------------------------------------ articulation and task targets
    @property
    def articulated_joints(self) -> list[str]:
        """Non-free joints of the scene's fixtures (drawer slides, stove knob): the fixture state goals read."""
        if not hasattr(self, "_art"):
            m = self.inner.sim.model
            out = []
            for name in sorted(self.inner.fixtures_dict):
                for j in self.inner.fixtures_dict[name].joints:
                    if int(m.jnt_type[m.joint_name2id(j)]) != 0:  # 0 = mjJNT_FREE
                        out.append(j)
            self._art = sorted(out)
        return self._art

    def joint_positions(self) -> np.ndarray:
        d = self.inner.sim.data
        return np.array([float(d.get_joint_qpos(j)) for j in self.articulated_joints], dtype=np.float64)

    def task_targets(self, task_id: int) -> dict:
        """What counts as target progress vs. off-target activity for `task_id` (any suite task: they share the
        scene). Derived from the BDDL goal predicates and the compiled articulation, never hand-listed:
          manipulated_objects  movable objects that are the first argument of a binary goal predicate
          touch_entities       manipulated objects + entities of unary-predicate arguments (the cabinet of
                               "open <drawer region>", the stove of "turnon")
          goal_joints          articulated joints of goal arguments (the drawer of a drawer region, the knob
                               of the stove of "turnon")
          own_entities         the task's entities (objects of interest; destinations included)
          non_target_objects   movable objects outside own_entities
          non_target_joints    articulated joints that are not goal joints
        """
        parsed = self._parsed[task_id]
        sites, art = self.inner.object_sites_dict, set(self.articulated_joints)
        manipulated, touch, joints = set(), set(), set()
        for pred, *args in parsed["goal_state"]:
            if len(args) == 2 and args[0] in self.movable_objects:
                manipulated.add(args[0])
                touch.add(args[0])
            if len(args) == 1:
                touch.add(entity_of(args[0], parsed))
            for a in args:
                js = sites[a].joints if a in sites else self.inner.get_object(a).joints
                if len(args) == 1 or a in sites:
                    joints |= set(js or []) & art
        own = task_entities(parsed)
        return {"task_id": task_id, "manipulated_objects": sorted(manipulated), "touch_entities": sorted(touch),
                "goal_joints": sorted(joints), "own_entities": sorted(own),
                "non_target_objects": sorted(set(self.movable_objects) - own),
                "non_target_joints": sorted(art - joints)}

    # ------------------------------------------------------------------ masks
    def _instances(self) -> list[str]:
        if not self.segmentation:
            raise RuntimeError("segmentation disabled")
        return list(self._env.env.model.instances_to_ids.keys())

    def instance_ids(self) -> dict[str, int]:
        """Instance name -> id in the instance-segmentation image: enumeration index + 1 (0 = no instance),
        LIBERO's convention. Computed here for every instance, because LIBERO's own mapping looks for a robot
        named "Panda0", which tabletop scenes call "MountedPanda0": it then maps the arm like an object and
        leaves the mount and gripper unmapped. Pixel-level correctness is tested in tests/test_libero_env.py."""
        return {name: i + 1 for i, name in enumerate(self._instances())}

    def robot_segmentation_ids(self) -> set[int]:
        """Ids of every instance that is not a BDDL object or fixture of the loaded scene (the arm, its mount, the
        gripper)."""
        import libero.libero.envs.bddl_utils as BDDLUtils

        if not hasattr(self, "_loaded_entities"):
            p = BDDLUtils.robosuite_parse_problem(self.bddl_file)
            self._loaded_entities = {n for group in (p["objects"], p["fixtures"]) for names in group.values() for n in names}
        entities = self._loaded_entities
        robot = {i for name, i in self.instance_ids().items() if name not in entities}
        if not robot:
            raise RuntimeError("no robot instances found")
        return robot

    def masks(self, frame: Frame) -> dict:
        """Per-camera boolean masks from instance segmentation (same orientation as the policy images).

        background = pixels belonging to neither the robot nor any protected entity (union over all suite
                     tasks; see `protected_entities`). Table, floor, walls and unlabeled pixels are background.
        target     = pixels of the instructed task's own entities (`own_entities`).
        """
        if frame.agentview_seg is None:
            raise RuntimeError("frame has no segmentation (create the env with segmentation=True)")
        ids = self.instance_ids()
        missing = [e for e in self.protected_entities if e not in ids]
        if missing:
            raise RuntimeError(f"protected entities without a segmentation id: {missing}")
        protected = np.array(sorted({ids[e] for e in self.protected_entities} | self.robot_segmentation_ids()))
        own = np.array(sorted(ids[e] for e in self.own_entities))
        # anti-aliased segmentation mislabels boundary pixels: protect a band around every protected pixel
        # (protocol corruption.protected_mask_dilation_px)
        from scipy import ndimage

        from vlamech.config import protocol

        band = int(protocol("corruption.protected_mask_dilation_px"))
        out = {}
        for cam, seg in (("agentview", frame.agentview_seg), ("wrist", frame.wrist_seg)):
            prot = np.isin(seg, protected)
            if band > 0:
                prot = ndimage.binary_dilation(prot, structure=np.ones((3, 3), bool), iterations=band)
            out[cam] = {"background": ~prot, "target": np.isin(seg, own)}
        return out

    def close(self):
        self._env.close()


def _body_chain(model, b: int) -> list[int]:
    """Body b and all its ancestors up to (excluding) the world body."""
    chain = []
    while b > 0:
        chain.append(b)
        b = int(model.body_parentid[b])
    return chain
