#!/usr/bin/env python3
"""
Draw DKPrompt overview with Active Perception integration showing:
1. VQA query responses: Yes, No, Uncertain
2. Three distinct decision paths
3. Active perception for uncertain responses
4. World state update based on Yes/No outcomes
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import numpy as np

def create_dkprompt_with_active_perception():
    """Create DKPrompt overview with all three VQA decision paths."""

    fig = plt.figure(figsize=(22, 14))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis('off')

    fig.suptitle('DKPrompt: Complete VQA Decision Flow with Active Perception',
                 fontsize=19, fontweight='bold', y=0.98)

    # ============================================================================
    # TOP: CONTEXT
    # ============================================================================

    context_box = FancyBboxPatch((1, 12.5), 20, 1.2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#1565C0', facecolor='#E3F2FD', linewidth=2.5)
    ax.add_patch(context_box)

    context_text = '''DKPrompt verifies each action precondition (BEFORE execution) and effect (AFTER execution) using VQA.
VLM responses have THREE outcomes: YES (satisfied), NO (unsatisfied), or UNCERTAIN (activate active perception).'''

    ax.text(11, 13.1, context_text, fontsize=10, ha='center', va='center',
            fontweight='bold', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # MAIN FLOW: PRECONDITION CHECKING (LEFT) vs EFFECT VERIFICATION (RIGHT)
    # ============================================================================

    # Left column: Precondition checking
    ax.text(4.5, 11.8, 'PRECONDITION CHECKING (Before Action)', fontsize=12, fontweight='bold',
            ha='center', color='#1565C0',
            bbox=dict(boxstyle='round', facecolor='#BBDEFB', alpha=0.7))

    # Right column: Effect verification
    ax.text(17.5, 11.8, 'EFFECT VERIFICATION (After Action)', fontsize=12, fontweight='bold',
            ha='center', color='#C62828',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.7))

    # ============================================================================
    # LEFT SIDE: PRECONDITION CHECKING FLOW
    # ============================================================================

    y_pre = 11

    # Step 1: Get current state
    pre_box1 = FancyBboxPatch((0.5, y_pre - 0.8), 3.5, 1,
                              boxstyle="round,pad=0.08",
                              edgecolor='#1565C0', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(pre_box1)
    ax.text(2.25, y_pre - 0.3, 'Current State\n& Observation', fontsize=9, fontweight='bold', ha='center')

    arrow_down = FancyArrowPatch((2.25, y_pre - 0.8), (2.25, y_pre - 1.4),
                                arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#1565C0')
    ax.add_patch(arrow_down)

    # Step 2: VQA Precondition Query
    y_pre -= 1.6
    pre_box2 = FancyBboxPatch((0.3, y_pre - 1.2), 3.9, 1.4,
                              boxstyle="round,pad=0.08",
                              edgecolor='#D32F2F', facecolor='#FFEBEE', linewidth=2.5)
    ax.add_patch(pre_box2)
    ax.text(2.25, y_pre - 0.3, 'VQA Query:\nPrecondition Check', fontsize=9, fontweight='bold', ha='center')
    ax.text(2.25, y_pre - 0.8, '(e.g., "Is gripper empty?")', fontsize=8, ha='center', style='italic')

    arrow_down = FancyArrowPatch((2.25, y_pre - 1.2), (2.25, y_pre - 1.8),
                                arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#D32F2F')
    ax.add_patch(arrow_down)

    # Decision point: VQA Response
    y_pre -= 2.2
    decision_circle = Circle((2.25, y_pre), 0.4, edgecolor='#F57C00', facecolor='#FFE0B2', linewidth=2.5)
    ax.add_patch(decision_circle)
    ax.text(2.25, y_pre, 'VQA?', fontsize=10, fontweight='bold', ha='center', va='center')

    # ============================================================================
    # PATH 1: YES (Left) - Precondition Satisfied
    # ============================================================================

    # YES path
    y_yes = y_pre - 0.8
    arrow_yes = FancyArrowPatch((1.4, y_pre - 0.35), (0.8, y_yes),
                               arrowstyle='->', mutation_scale=20, linewidth=2.5, color='green')
    ax.add_patch(arrow_yes)
    ax.text(0.8, y_pre - 0.5, 'YES', fontsize=9, fontweight='bold', ha='center', color='green',
           bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))

    yes_box = FancyBboxPatch((0.1, y_yes - 1), 3.4, 1,
                            boxstyle="round,pad=0.08",
                            edgecolor='green', facecolor='#C8E6C9', linewidth=2.5)
    ax.add_patch(yes_box)
    ax.text(1.8, y_yes - 0.5, '✓ SATISFIED\nProceed to Execute', fontsize=9, fontweight='bold', ha='center')

    # ============================================================================
    # PATH 2: NO (Center-Left) - Precondition Unsatisfied
    # ============================================================================

    y_no = y_pre - 0.8
    arrow_no = FancyArrowPatch((2.25, y_pre - 0.4), (2.25, y_no - 0.2),
                              arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#D32F2F')
    ax.add_patch(arrow_no)
    ax.text(2.6, y_pre - 0.6, 'NO', fontsize=9, fontweight='bold', ha='center', color='#D32F2F',
           bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))

    no_box = FancyBboxPatch((0.8, y_no - 1.2), 2.9, 1.3,
                           boxstyle="round,pad=0.08",
                           edgecolor='#D32F2F', facecolor='#FFCDD2', linewidth=2.5)
    ax.add_patch(no_box)
    ax.text(2.25, y_no - 0.4, '✗ UNSATISFIED\nCannot Execute', fontsize=9, fontweight='bold', ha='center')
    ax.text(2.25, y_no - 0.85, '→ Update PDDL\n→ Replan', fontsize=8, ha='center', style='italic')

    # ============================================================================
    # PATH 3: UNCERTAIN (Center-Right) - Active Perception
    # ============================================================================

    y_uncertain = y_pre - 0.8
    arrow_uncertain = FancyArrowPatch((3.1, y_pre - 0.35), (3.7, y_uncertain),
                                     arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#F57C00')
    ax.add_patch(arrow_uncertain)
    ax.text(3.9, y_pre - 0.5, 'UNCERTAIN', fontsize=9, fontweight='bold', ha='center', color='#F57C00',
           bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.8))

    uncertain_main_box = FancyBboxPatch((3.3, y_uncertain - 3), 3, 2.9,
                                        boxstyle="round,pad=0.1",
                                        edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2.5)
    ax.add_patch(uncertain_main_box)

    ax.text(4.8, y_uncertain - 0.3, 'ACTIVE PERCEPTION', fontsize=10, fontweight='bold', ha='center', color='#F57C00')

    uncertain_steps = '''1. Identify target object
   from predicate

2. Sample viewpoints
   around object

3. Navigate to new
   viewpoint

4. Capture new
   observation

5. Re-query VLM
   with new view'''

    ax.text(4.8, y_uncertain - 1.6, uncertain_steps, fontsize=7.5, ha='center', va='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Arrow looping back from active perception
    arrow_loop = FancyArrowPatch((3.3, y_uncertain - 1), (2.5, y_pre - 0.2),
                               arrowstyle='->', mutation_scale=20, linewidth=2.5,
                               color='#F57C00', linestyle='dashed',
                               connectionstyle="arc3,rad=-0.5")
    ax.add_patch(arrow_loop)
    ax.text(2.2, y_pre + 0.5, 'Get confident\nanswer', fontsize=7.5, ha='center', style='italic', color='#F57C00')

    # ============================================================================
    # RIGHT SIDE: EFFECT VERIFICATION FLOW
    # ============================================================================

    y_eff = 11

    # Step 1: Action executed
    eff_box1 = FancyBboxPatch((15.5, y_eff - 0.8), 3.5, 1,
                              boxstyle="round,pad=0.08",
                              edgecolor='#558B2F', facecolor='#DCEDC8', linewidth=2)
    ax.add_patch(eff_box1)
    ax.text(17.25, y_eff - 0.3, 'Action Executed\nby Robot', fontsize=9, fontweight='bold', ha='center')

    arrow_down = FancyArrowPatch((17.25, y_eff - 0.8), (17.25, y_eff - 1.4),
                                arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#558B2F')
    ax.add_patch(arrow_down)

    # Step 2: New observation
    y_eff -= 1.6
    eff_box2 = FancyBboxPatch((15.3, y_eff - 1), 4, 1.2,
                              boxstyle="round,pad=0.08",
                              edgecolor='#1565C0', facecolor='#E3F2FD', linewidth=2)
    ax.add_patch(eff_box2)
    ax.text(17.25, y_eff - 0.3, 'New Observation\nfrom Robot Sensors', fontsize=9, fontweight='bold', ha='center')

    arrow_down = FancyArrowPatch((17.25, y_eff - 1), (17.25, y_eff - 1.6),
                                arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#1565C0')
    ax.add_patch(arrow_down)

    # Step 3: VQA Effect Query
    y_eff -= 1.8
    eff_box3 = FancyBboxPatch((15.1, y_eff - 1.2), 4.4, 1.4,
                              boxstyle="round,pad=0.08",
                              edgecolor='#D32F2F', facecolor='#FFEBEE', linewidth=2.5)
    ax.add_patch(eff_box3)
    ax.text(17.25, y_eff - 0.3, 'VQA Query:\nEffect Verification', fontsize=9, fontweight='bold', ha='center')
    ax.text(17.25, y_eff - 0.8, '(e.g., "Is bottle in gripper?")', fontsize=8, ha='center', style='italic')

    arrow_down = FancyArrowPatch((17.25, y_eff - 1.2), (17.25, y_eff - 1.8),
                                arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#D32F2F')
    ax.add_patch(arrow_down)

    # Decision point: VQA Response (right side)
    y_eff -= 2.2
    decision_circle_r = Circle((17.25, y_eff), 0.4, edgecolor='#F57C00', facecolor='#FFE0B2', linewidth=2.5)
    ax.add_patch(decision_circle_r)
    ax.text(17.25, y_eff, 'VQA?', fontsize=10, fontweight='bold', ha='center', va='center')

    # ============================================================================
    # RIGHT PATH 1: YES (Effect Match)
    # ============================================================================

    y_eff_yes = y_eff - 0.8
    arrow_eff_yes = FancyArrowPatch((16.4, y_eff - 0.35), (15.8, y_eff_yes),
                                   arrowstyle='->', mutation_scale=20, linewidth=2.5, color='green')
    ax.add_patch(arrow_eff_yes)
    ax.text(15.8, y_eff - 0.5, 'YES', fontsize=9, fontweight='bold', ha='center', color='green',
           bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))

    eff_yes_box = FancyBboxPatch((14.9, y_eff_yes - 1), 3.4, 1,
                                boxstyle="round,pad=0.08",
                                edgecolor='green', facecolor='#C8E6C9', linewidth=2.5)
    ax.add_patch(eff_yes_box)
    ax.text(16.6, y_eff_yes - 0.5, '✓ SATISFIED\nContinue to Next', fontsize=9, fontweight='bold', ha='center')

    # ============================================================================
    # RIGHT PATH 2: NO (Effect Mismatch - SITUATION DETECTED)
    # ============================================================================

    y_eff_no = y_eff - 0.8
    arrow_eff_no = FancyArrowPatch((17.25, y_eff - 0.4), (17.25, y_eff_no - 0.2),
                                  arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#D32F2F')
    ax.add_patch(arrow_eff_no)
    ax.text(17.9, y_eff - 0.6, 'NO', fontsize=9, fontweight='bold', ha='center', color='#D32F2F',
           bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))

    eff_no_box = FancyBboxPatch((15.6, y_eff_no - 1.2), 2.9, 1.3,
                               boxstyle="round,pad=0.08",
                               edgecolor='#D32F2F', facecolor='#FFCDD2', linewidth=2.5)
    ax.add_patch(eff_no_box)
    ax.text(17.25, y_eff_no - 0.4, '✗ SITUATION!', fontsize=9, fontweight='bold', ha='center')
    ax.text(17.25, y_eff_no - 0.85, '→ Update PDDL\n→ Replan', fontsize=8, ha='center', style='italic')

    # ============================================================================
    # RIGHT PATH 3: UNCERTAIN (Effect Uncertain - Active Perception)
    # ============================================================================

    y_eff_uncertain = y_eff - 0.8
    arrow_eff_uncertain = FancyArrowPatch((18.1, y_eff - 0.35), (18.7, y_eff_uncertain),
                                         arrowstyle='->', mutation_scale=20, linewidth=2.5, color='#F57C00')
    ax.add_patch(arrow_eff_uncertain)
    ax.text(18.9, y_eff - 0.5, 'UNCERTAIN', fontsize=9, fontweight='bold', ha='center', color='#F57C00',
           bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.8))

    eff_uncertain_box = FancyBboxPatch((18.3, y_eff_uncertain - 2.8), 3, 2.7,
                                       boxstyle="round,pad=0.1",
                                       edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2.5)
    ax.add_patch(eff_uncertain_box)

    ax.text(19.8, y_eff_uncertain - 0.3, 'ACTIVE PERCEPTION', fontsize=10, fontweight='bold', ha='center', color='#F57C00')

    eff_uncertain_steps = '''1. Identify target
   object

2. Sample
   viewpoints

3. Navigate to
   better view

4. Capture new
   RGB image

5. Re-query VLM'''

    ax.text(19.8, y_eff_uncertain - 1.5, eff_uncertain_steps, fontsize=7.5, ha='center', va='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Arrow looping back
    arrow_eff_loop = FancyArrowPatch((18.3, y_eff_uncertain - 1), (17.9, y_eff - 0.2),
                                   arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                   color='#F57C00', linestyle='dashed',
                                   connectionstyle="arc3,rad=0.5")
    ax.add_patch(arrow_eff_loop)
    ax.text(18.8, y_eff + 0.5, 'Get confident\nanswer', fontsize=7.5, ha='center', style='italic', color='#F57C00')

    # ============================================================================
    # BOTTOM: SUMMARY TABLE
    # ============================================================================

    summary_y = 0.8

    # Summary box
    summary_box = FancyBboxPatch((0.5, summary_y - 2.5), 21, 2.4,
                                 boxstyle="round,pad=0.15",
                                 edgecolor='#000000', facecolor='#F5F5F5', linewidth=2)
    ax.add_patch(summary_box)

    ax.text(11, summary_y - 0.2, 'VQA RESPONSE DECISION TABLE', fontsize=11, fontweight='bold', ha='center')

    # Table header
    headers_y = summary_y - 0.6
    ax.text(2, headers_y, 'Response', fontsize=9, fontweight='bold', ha='left',
           bbox=dict(boxstyle='round', facecolor='#BBDEFB', alpha=0.8))
    ax.text(5.5, headers_y, 'Precondition Path', fontsize=9, fontweight='bold', ha='left',
           bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
    ax.text(11, headers_y, 'Effect Path', fontsize=9, fontweight='bold', ha='left',
           bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
    ax.text(17, headers_y, 'Action Required', fontsize=9, fontweight='bold', ha='left',
           bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.8))

    # Row 1: YES
    row1_y = headers_y - 0.5
    ax.text(2, row1_y, 'YES\n(Matches expected)', fontsize=8, fontweight='bold', ha='left',
           bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))
    ax.text(5.5, row1_y, 'Precondition satisfied\n→ Execute action', fontsize=8, ha='left', family='monospace')
    ax.text(11, row1_y, 'Effect satisfied\n→ Continue normally', fontsize=8, ha='left', family='monospace')
    ax.text(17, row1_y, 'No state update', fontsize=8, ha='left', style='italic', color='green', fontweight='bold')

    # Row 2: NO
    row2_y = row1_y - 0.5
    ax.text(2, row2_y, 'NO\n(Mismatches expected)', fontsize=8, fontweight='bold', ha='left',
           bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))
    ax.text(5.5, row2_y, 'Cannot execute\n→ Update PDDL', fontsize=8, ha='left', family='monospace')
    ax.text(11, row2_y, 'Situation detected\n→ Update PDDL', fontsize=8, ha='left', family='monospace')
    ax.text(17, row2_y, 'Update world state\n→ Call planner', fontsize=8, ha='left', style='italic', color='#D32F2F', fontweight='bold')

    # Row 3: UNCERTAIN
    row3_y = row2_y - 0.5
    ax.text(2, row3_y, 'UNCERTAIN\n(VLM not confident)', fontsize=8, fontweight='bold', ha='left',
           bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.8))
    ax.text(5.5, row3_y, 'Trigger active perception\nNavigate → Re-query', fontsize=8, ha='left', family='monospace')
    ax.text(11, row3_y, 'Trigger active perception\nNavigate → Re-query', fontsize=8, ha='left', family='monospace')
    ax.text(17, row3_y, 'NO state update yet\nWait for confident answer', fontsize=8, ha='left', style='italic', color='#F57C00', fontweight='bold')

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("Creating DKPrompt with Active Perception Overview...")
    fig = create_dkprompt_with_active_perception()
    fig.savefig('/home/aoloo/code/vlm-tamp/dkprompt_with_active_perception_complete.png',
                dpi=300, bbox_inches='tight')
    print("✅ Saved: dkprompt_with_active_perception_complete.png")

    print("\n" + "="*70)
    print("DIAGRAM CREATED SUCCESSFULLY")
    print("="*70)
    print("\nThis diagram shows:")
    print("  • LEFT: Precondition checking with three decision paths")
    print("  • RIGHT: Effect verification with three decision paths")
    print("  • BOTTOM: Summary table of all three response types")
    print("\nKey insight:")
    print("  - YES: Proceed normally, no state update")
    print("  - NO: Situation detected, update PDDL and replan")
    print("  - UNCERTAIN: Activate active perception for better view")
    print("                Then proceed with confident Yes/No answer")
