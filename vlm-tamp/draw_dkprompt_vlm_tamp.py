#!/usr/bin/env python3
"""
Draw DKPrompt VLM-TAMP Overview showing:
1. VQA-based querying of domain knowledge (preconditions and effects)
2. Classical planner integration with updated world states
3. Left side: Precondition checking before action execution
4. Right side: Effect verification after action execution
5. Replanning trigger mechanism
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
import numpy as np

def create_dkprompt_overview():
    """Create comprehensive DKPrompt VLM-TAMP overview."""

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')

    fig.suptitle('DKPrompt: VQA-Based Planning with Domain Knowledge Integration',
                 fontsize=18, fontweight='bold', y=0.98)

    # ============================================================================
    # TOP: MAIN CONCEPT
    # ============================================================================

    concept_box = FancyBboxPatch((1, 10.5), 18, 1.2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#D32F2F', facecolor='#FFEBEE', linewidth=2.5)
    ax.add_patch(concept_box)

    concept_text = '''VQA-Based Planning: Query robot's current observation against domain knowledge (action preconditions and effects)
→ Generate new valid plan using updated world states → Replanning triggered when conditions unsatisfied'''

    ax.text(10, 11.1, concept_text, fontsize=10, ha='center', va='center',
            fontweight='bold', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # SECTION 1: LEFT SIDE - PRECONDITION CHECKING
    # ============================================================================

    ax.text(5, 10, 'LEFT: PRECONDITION CHECKING (Before Action Execution)',
            fontsize=12, fontweight='bold', ha='center', color='#1565C0')

    # Step 1: Current State
    y = 9.3
    state_box = FancyBboxPatch((0.5, y - 0.8), 3.5, 1,
                               boxstyle="round,pad=0.08",
                               edgecolor='#1565C0', facecolor='#E3F2FD', linewidth=2.5)
    ax.add_patch(state_box)
    ax.text(2.25, y - 0.3, 'Current Observation\n(World State)', fontsize=10, fontweight='bold', ha='center')

    arrow_down1 = FancyArrowPatch((2.25, y - 0.8), (2.25, y - 1.4),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#1565C0')
    ax.add_patch(arrow_down1)

    # Step 2: VQA Precondition Query
    y -= 1.6
    vqa_pre_box = FancyBboxPatch((0.2, y - 1.4), 4.1, 1.6,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#D32F2F', facecolor='#FFEBEE', linewidth=2.5)
    ax.add_patch(vqa_pre_box)
    ax.text(2.25, y - 0.2, 'VQA Precondition Queries', fontsize=10, fontweight='bold', ha='center')

    vqa_text = '''For action: "pickup(bottle)"
Query predicates:
• "Is gripper empty?"
• "Is bottle on table?"
• "Is bottle in view?"'''

    ax.text(2.25, y - 0.9, vqa_text, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    arrow_down2 = FancyArrowPatch((2.25, y - 1.4), (2.25, y - 2),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#D32F2F')
    ax.add_patch(arrow_down2)

    # Step 3: Domain Knowledge Matching
    y -= 2.2
    domain_pre_box = FancyBboxPatch((0.2, y - 1.2), 4.1, 1.4,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='#00796B', facecolor='#B2DFDB', linewidth=2.5)
    ax.add_patch(domain_pre_box)
    ax.text(2.25, y - 0.2, 'Domain Knowledge\n(Action Preconditions)', fontsize=10, fontweight='bold', ha='center')

    domain_text = '''PDDL Definition:
(:action pickup
 :preconditions
   (gripper_empty ?g)
   (object_at ?obj ?loc)
   (visible ?obj))'''

    ax.text(2.25, y - 0.85, domain_text, fontsize=7.5, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    arrow_down3 = FancyArrowPatch((2.25, y - 1.2), (2.25, y - 1.8),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#00796B')
    ax.add_patch(arrow_down3)

    # Step 4: Preconditions Verification
    y -= 2
    verify_pre_box = FancyBboxPatch((0.2, y - 1.2), 4.1, 1.4,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='#F57C00', facecolor='#FFE0B2', linewidth=2.5)
    ax.add_patch(verify_pre_box)
    ax.text(2.25, y - 0.2, 'Precondition Verification', fontsize=10, fontweight='bold', ha='center')

    verify_pre_text = '''VQA Responses:
✓ gripper_empty: TRUE
✓ object_at: TRUE
✓ visible: TRUE

Result: ALL SATISFIED'''

    ax.text(2.25, y - 0.85, verify_pre_text, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))

    arrow_down4 = FancyArrowPatch((2.25, y - 1.2), (2.25, y - 1.8),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#F57C00')
    ax.add_patch(arrow_down4)

    # Step 5: Execute Action
    y -= 2
    execute_box = FancyBboxPatch((0.2, y - 0.8), 4.1, 1,
                                 boxstyle="round,pad=0.08",
                                 edgecolor='#558B2F', facecolor='#DCEDC8', linewidth=2.5)
    ax.add_patch(execute_box)
    ax.text(2.25, y - 0.4, 'EXECUTE ACTION\n(Preconditions OK)', fontsize=10, fontweight='bold', ha='center')

    # ============================================================================
    # SECTION 2: RIGHT SIDE - EFFECT VERIFICATION
    # ============================================================================

    ax.text(15, 10, 'RIGHT: EFFECT VERIFICATION (After Action Execution)',
            fontsize=12, fontweight='bold', ha='center', color='#C62828')

    # Step 1: Action Executed
    y = 9.3
    executed_box = FancyBboxPatch((12, y - 0.8), 3.5, 1,
                                  boxstyle="round,pad=0.08",
                                  edgecolor='#558B2F', facecolor='#DCEDC8', linewidth=2.5)
    ax.add_patch(executed_box)
    ax.text(13.75, y - 0.3, 'Action Executed\n(Robot Manipulates)', fontsize=10, fontweight='bold', ha='center')

    arrow_down5 = FancyArrowPatch((13.75, y - 0.8), (13.75, y - 1.4),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#558B2F')
    ax.add_patch(arrow_down5)

    # Step 2: New Observation
    y -= 1.6
    new_obs_box = FancyBboxPatch((11.7, y - 1), 4.1, 1.2,
                                 boxstyle="round,pad=0.08",
                                 edgecolor='#1565C0', facecolor='#E3F2FD', linewidth=2.5)
    ax.add_patch(new_obs_box)
    ax.text(13.75, y - 0.3, 'New Observation\n(Robot Perception)', fontsize=10, fontweight='bold', ha='center')
    ax.text(13.75, y - 0.7, 'RGB-D image → VQA', fontsize=8, ha='center', style='italic')

    arrow_down6 = FancyArrowPatch((13.75, y - 1), (13.75, y - 1.6),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#1565C0')
    ax.add_patch(arrow_down6)

    # Step 3: VQA Effect Query
    y -= 1.8
    vqa_eff_box = FancyBboxPatch((11.4, y - 1.4), 4.7, 1.6,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#D32F2F', facecolor='#FFEBEE', linewidth=2.5)
    ax.add_patch(vqa_eff_box)
    ax.text(13.75, y - 0.2, 'VQA Effect Queries', fontsize=10, fontweight='bold', ha='center')

    vqa_eff_text = '''For action: "pickup(bottle)"
Query predicates:
• "Is gripper holding bottle?"
• "Is bottle no longer on table?"
• "Is gripper NOT empty?"'''

    ax.text(13.75, y - 0.9, vqa_eff_text, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    arrow_down7 = FancyArrowPatch((13.75, y - 1.4), (13.75, y - 2),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#D32F2F')
    ax.add_patch(arrow_down7)

    # Step 4: Domain Knowledge Matching (Effects)
    y -= 2.2
    domain_eff_box = FancyBboxPatch((11.4, y - 1.2), 4.7, 1.4,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='#00796B', facecolor='#B2DFDB', linewidth=2.5)
    ax.add_patch(domain_eff_box)
    ax.text(13.75, y - 0.2, 'Domain Knowledge\n(Action Effects)', fontsize=10, fontweight='bold', ha='center')

    domain_eff_text = '''PDDL Definition:
(:action pickup
 :effect
   (holding ?gripper ?obj)
   (not (object_at ?obj ?loc))
   (not (gripper_empty ?g)))'''

    ax.text(13.75, y - 0.85, domain_eff_text, fontsize=7.5, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    arrow_down8 = FancyArrowPatch((13.75, y - 1.2), (13.75, y - 1.8),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#00796B')
    ax.add_patch(arrow_down8)

    # Step 5: Effects Verification
    y -= 2
    verify_eff_box = FancyBboxPatch((11.4, y - 1.2), 4.7, 1.4,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='#F57C00', facecolor='#FFE0B2', linewidth=2.5)
    ax.add_patch(verify_eff_box)
    ax.text(13.75, y - 0.2, 'Effect Verification', fontsize=10, fontweight='bold', ha='center')

    verify_eff_text = '''VQA Responses:
✓ holding: TRUE
✓ not_at_table: TRUE
✓ not_gripper_empty: TRUE

Result: ALL SATISFIED'''

    ax.text(13.75, y - 0.85, verify_eff_text, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))

    arrow_down9 = FancyArrowPatch((13.75, y - 1.2), (13.75, y - 1.8),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#F57C00')
    ax.add_patch(arrow_down9)

    # Step 6: Update World State
    y -= 2
    update_box = FancyBboxPatch((11.4, y - 0.8), 4.7, 1,
                                boxstyle="round,pad=0.08",
                                edgecolor='#558B2F', facecolor='#DCEDC8', linewidth=2.5)
    ax.add_patch(update_box)
    ax.text(13.75, y - 0.4, 'UPDATE WORLD STATE\n(Effects Confirmed)', fontsize=10, fontweight='bold', ha='center')

    # ============================================================================
    # CENTER: REPLANNING DECISION LOGIC
    # ============================================================================

    center_y = 3.5
    replan_main = FancyBboxPatch((4, center_y - 3), 12, 2.8,
                                 boxstyle="round,pad=0.15",
                                 edgecolor='#6A1B9A', facecolor='#F3E5F5', linewidth=3)
    ax.add_patch(replan_main)

    ax.text(10, center_y - 0.2, 'REPLANNING DECISION LOGIC',
            fontsize=12, fontweight='bold', ha='center', color='#6A1B9A')

    replan_logic = '''Preconditions Unsatisfied?
  ├─ YES → CANNOT EXECUTE
  │         Update PDDL state with actual observations
  │         Call planner to generate NEW PLAN
  │         Execute recovery actions
  │
  └─ NO  → Continue with current plan

Effects Unsatisfied (After Execution)?
  ├─ YES → ACTION FAILED
  │         Update PDDL state with actual world state
  │         Planner recognizes failure
  │         Triggers REPLANNING with updated knowledge
  │
  └─ NO  → Action successful, proceed to next action

Both Satisfied?
  └─ Continue execution of next planned action'''

    ax.text(10, center_y - 1.7, replan_logic, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # ============================================================================
    # BOTTOM: CONNECTING ARROWS & FLOW
    # ============================================================================

    # Arrow from left preconditions to center
    arrow_left_center = FancyArrowPatch((4.3, 2.5), (7, center_y - 0.2),
                                       arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                       color='#1565C0', linestyle='dashed')
    ax.add_patch(arrow_left_center)
    ax.text(5.2, 2.9, 'Preconditions', fontsize=8, ha='center', style='italic', color='#1565C0')

    # Arrow from right effects to center
    arrow_right_center = FancyArrowPatch((15.7, 2.5), (13, center_y - 0.2),
                                        arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                        color='#C62828', linestyle='dashed')
    ax.add_patch(arrow_right_center)
    ax.text(14.8, 2.9, 'Effects', fontsize=8, ha='center', style='italic', color='#C62828')

    # ============================================================================
    # BOTTOM LEFT: PREDICATE-ONLY QUERIES
    # ============================================================================

    pred_box = FancyBboxPatch((0.5, -0.5), 4.5, 1.2,
                              boxstyle="round,pad=0.1",
                              edgecolor='#00796B', facecolor='#E0F2F1', linewidth=2)
    ax.add_patch(pred_box)

    ax.text(2.75, 0.45, 'Note: Predicate-Only Queries',
            fontsize=10, fontweight='bold', ha='center', color='#00796B')

    pred_text = '''DKPrompt queries ONLY predicates:
  • gripper_empty(g): Boolean
  • object_at(obj, loc): Boolean
  • holding(g, obj): Boolean

NOT full scene understanding,
but specific state predicates'''

    ax.text(2.75, -0.1, pred_text, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # BOTTOM CENTER: CLASSICAL PLANNER INTEGRATION
    # ============================================================================

    planner_box = FancyBboxPatch((7.5, -0.5), 5, 1.2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#1565C0', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(planner_box)

    ax.text(10, 0.45, 'Classical Planner Integration',
            fontsize=10, fontweight='bold', ha='center', color='#1565C0')

    planner_text = '''When preconditions/effects fail:
  1. Update PDDL problem file
  2. Feed to classical planner
  3. Generate NEW valid plan
  4. Execute recovery actions

Example: Fast-Downward Planner'''

    ax.text(10, -0.1, planner_text, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # BOTTOM RIGHT: WORLD STATE UPDATES
    # ============================================================================

    state_update_box = FancyBboxPatch((15, -0.5), 4.5, 1.2,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='#F57C00', facecolor='#FFE0B2', linewidth=2)
    ax.add_patch(state_update_box)

    ax.text(17.25, 0.45, 'Updated World States',
            fontsize=10, fontweight='bold', ha='center', color='#F57C00')

    state_text = '''World state = Collection of predicates
Updated from VQA responses:
  • (gripper_empty g1): changed
  • (holding g1 bottle): added
  • (object_at bottle t1): removed

Directly feeds to planner'''

    ax.text(17.25, -0.1, state_text, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig

def create_example_scenario():
    """Create example scenario showing DKPrompt in action."""

    fig = plt.figure(figsize=(18, 11))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')

    fig.suptitle('Example Scenario: Egg Halving Task with Precondition & Effect Verification',
                 fontsize=16, fontweight='bold', y=0.97)

    # ============================================================================
    # TOP: INITIAL STATE
    # ============================================================================

    y = 10
    ax.text(9, y, 'INITIAL PLAN: [pickup(knife), cut_egg(knife), verify_egg()]',
            fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

    # ============================================================================
    # STEP 1: PRECONDITION CHECK FOR PICKUP
    # ============================================================================

    y -= 1.2
    ax.text(3, y + 0.3, 'STEP 1: Check Preconditions for pickup(knife)',
            fontsize=11, fontweight='bold', ha='center', color='#1565C0')

    y -= 0.6
    pre_check_box = FancyBboxPatch((0.3, y - 1.8), 5.4, 1.8,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='#1565C0', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(pre_check_box)

    pre_check = '''Preconditions for pickup(knife):
  1. gripper_empty(?g) → VQA: "Is gripper empty?"
     Response: TRUE ✓
  2. object_at(knife, kitchen) → VQA: "Is knife on kitchen counter?"
     Response: TRUE ✓
  3. reachable(knife) → VQA: "Can arm reach knife?"
     Response: TRUE ✓

ALL PRECONDITIONS SATISFIED → EXECUTE pickup(knife)'''

    ax.text(3, y - 1.2, pre_check, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # STEP 2: ACTION EXECUTION
    # ============================================================================

    y -= 2.2
    ax.text(3, y + 0.3, 'STEP 2: Execute pickup(knife)',
            fontsize=11, fontweight='bold', ha='center', color='#558B2F')

    y -= 0.6
    exec_box = FancyBboxPatch((0.3, y - 0.8), 5.4, 0.8,
                              boxstyle="round,pad=0.08",
                              edgecolor='#558B2F', facecolor='#DCEDC8', linewidth=2)
    ax.add_patch(exec_box)
    ax.text(3, y - 0.4, 'Robot: Open gripper → Move to knife → Close gripper → Retract',
            fontsize=9, ha='center', va='center', family='monospace')

    # ============================================================================
    # STEP 3: EFFECT VERIFICATION
    # ============================================================================

    y -= 1.2
    ax.text(3, y + 0.3, 'STEP 3: Verify Effects of pickup(knife)',
            fontsize=11, fontweight='bold', ha='center', color='#C62828')

    y -= 0.6
    effect_check_box = FancyBboxPatch((0.3, y - 1.8), 5.4, 1.8,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='#C62828', facecolor='#FFEBEE', linewidth=2)
    ax.add_patch(effect_check_box)

    effect_check = '''Expected Effects:
  1. holding(gripper, knife) → VQA: "Is knife in gripper?"
     Response: TRUE ✓
  2. ¬object_at(knife, kitchen) → VQA: "Is knife no longer on counter?"
     Response: TRUE ✓
  3. ¬gripper_empty(gripper) → VQA: "Is gripper occupied?"
     Response: TRUE ✓

ALL EFFECTS SATISFIED → World state updated → Continue to next action'''

    ax.text(3, y - 1.2, effect_check, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # PARALLEL: CUT EGG SCENARIO (RIGHT SIDE)
    # ============================================================================

    y_right = 10
    ax.text(15, y_right, 'SCENARIO: Egg Cutting Fails',
            fontsize=11, fontweight='bold', ha='center', color='#D32F2F',
            bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))

    y_right -= 1.2
    ax.text(15, y_right + 0.3, 'Expected: Both egg halves on plate',
            fontsize=10, fontweight='bold', ha='center', color='#D32F2F')

    y_right -= 0.6
    situation_box = FancyBboxPatch((9.3, y_right - 2.5), 5.4, 2.5,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='#D32F2F', facecolor='#FFCDD2', linewidth=2.5)
    ax.add_patch(situation_box)

    situation = '''Current Plan:
  [pickup(knife), cut_egg(knife), verify_egg()]

After cut_egg(knife) execution:

VQA Effect Queries:
  1. "Are both halves on plate?"
     Response: NO ✗ (One half fell!)
  2. "Is egg properly halved?"
     Response: PARTIALLY ✗

EFFECTS NOT SATISFIED!
→ FAILURE DETECTED → TRIGGER REPLANNING'''

    ax.text(15, y_right - 1.5, situation, fontsize=8, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # REPLANNING PROCESS
    # ============================================================================

    y_right -= 2.8
    ax.text(15, y_right + 0.3, 'REPLANNING TRIGGERED',
            fontsize=11, fontweight='bold', ha='center', color='#F57C00')

    y_right -= 0.6
    replan_process = FancyBboxPatch((9.3, y_right - 2.2), 5.4, 2.2,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='#F57C00', facecolor='#FFE0B2', linewidth=2)
    ax.add_patch(replan_process)

    replan_steps = '''1. Update PDDL World State:
   - egg_half_1(fallen): TRUE
   - egg_on_plate(partial): FALSE
   - need_recovery: TRUE

2. Call Classical Planner:
   - Input: Updated PDDL problem
   - Output: NEW valid plan

3. Generated Recovery Plan:
   [wait(60s), pickup(fallen_half),
    place_to_marker(fallen_half), verify_egg()]

4. Execute Recovery:
   - Pick fallen half
   - Place on plate
   - Verify success

5. Continue with updated state'''

    ax.text(15, y_right - 1.3, replan_steps, fontsize=7.5, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # CONNECTION ARROWS
    # ============================================================================

    # Arrow from Step 1 to Step 2
    arrow1 = FancyArrowPatch((3, 4.6), (3, 3.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#1565C0')
    ax.add_patch(arrow1)

    # Arrow from Step 2 to Step 3
    arrow2 = FancyArrowPatch((3, 3.8), (3, 2.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#558B2F')
    ax.add_patch(arrow2)

    # Arrow from left to right (failure scenario)
    arrow3 = FancyArrowPatch((5.7, 5.5), (9.3, 6.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5,
                            color='#D32F2F', linestyle='dashed')
    ax.add_patch(arrow3)
    ax.text(7, 6.5, 'Compared\nwith actual\noutcome', fontsize=8, ha='center', color='#D32F2F')

    # Arrow from situation to replanning
    arrow4 = FancyArrowPatch((15, 4.8), (15, 3.6),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#F57C00')
    ax.add_patch(arrow4)
    ax.text(15.8, 4.2, 'Failure\nDetected', fontsize=8, ha='left', color='#F57C00')

    # ============================================================================
    # BOTTOM: KEY INSIGHTS
    # ============================================================================

    insights = FancyBboxPatch((0.3, -0.8), 17.4, 0.75,
                              boxstyle="round,pad=0.08",
                              edgecolor='#000000', facecolor='#FFF9C4', linewidth=2)
    ax.add_patch(insights)

    insight_text = '''KEY INSIGHT: DKPrompt bridges VLM perception with classical planning by querying specific predicates about world state.
Preconditions checked BEFORE execution ensure feasibility. Effects checked AFTER execution detect failures. Both trigger replanning with updated world knowledge.'''

    ax.text(9, -0.4, insight_text, fontsize=9, ha='center', va='center',
            fontweight='bold', family='monospace')

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("Creating DKPrompt VLM-TAMP Overview...")
    fig1 = create_dkprompt_overview()
    fig1.savefig('/home/aoloo/code/vlm-tamp/dkprompt_vlm_tamp_overview.png',
                 dpi=300, bbox_inches='tight')
    print("✅ Saved: dkprompt_vlm_tamp_overview.png")

    print("\nCreating Example Scenario...")
    fig2 = create_example_scenario()
    fig2.savefig('/home/aoloo/code/vlm-tamp/dkprompt_example_scenario.png',
                 dpi=300, bbox_inches='tight')
    print("✅ Saved: dkprompt_example_scenario.png")

    print("\n" + "="*70)
    print("DKPrompt DIAGRAMS CREATED SUCCESSFULLY")
    print("="*70)
    print("\nOutput files:")
    print("  1. dkprompt_vlm_tamp_overview.png")
    print("     - VQA-based precondition checking (LEFT)")
    print("     - Effect verification after execution (RIGHT)")
    print("     - Replanning decision logic (CENTER)")
    print("")
    print("  2. dkprompt_example_scenario.png")
    print("     - Concrete example: egg halving task")
    print("     - Normal execution flow")
    print("     - Failure detection and replanning")
    print("\nThese diagrams show DKPrompt's core mechanism:")
    print("  • Query robot observation against domain knowledge")
    print("  • VQA tasks for preconditions and effects")
    print("  • Classical planner integration with updated states")
    print("  • Automatic replanning on failure")

    plt.show()
