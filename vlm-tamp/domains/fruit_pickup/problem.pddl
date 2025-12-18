;; PDDL Problem for Active Perception Testing: Fruit Pickup
;;
;; Scenario: Pick up specific red fruits from a table with multiple similar-colored objects
;; This tests active perception when VLM is uncertain about which red object is which
;;
;; Setup:
;;   Main table with 4 fruits clustered together:
;;     - apple1 (red apple)
;;     - tomato1 (red tomato)
;;     - strawberry1 (red strawberry)
;;     - orange1 (orange fruit - easier to distinguish)
;;
;;   Goal table where fruits should be placed
;;
;; Challenge:
;;   1. Three fruits are red (apple, tomato, strawberry) - hard to distinguish
;;   2. Fruits are close together on the table
;;   3. VLM may be uncertain from current viewpoint
;;   4. Active perception (head movement + base repositioning) needed to identify correctly
;;
;; Task: Pick the apple and tomato, place them on the goal table

(define (problem pickup_red_fruits)
  (:domain fruit_pickup)

  (:objects
    robot1 - robot

    ;; Fruits (3 are red - requires careful discrimination!)
    apple1 - fruit          ; Red apple (round, red)
    tomato1 - fruit         ; Red tomato (round, red, looks like apple!)
    strawberry1 - fruit     ; Red strawberry (smaller, textured)
    orange1 - fruit         ; Orange (control - easy to identify)

    ;; Locations
    loc_main_table - location
    loc_goal_table - location

    ;; Surfaces
    main_table - surface
    goal_table - surface
  )

  (:init
    ;; Robot starts at main table where fruits are
    (at robot1 loc_main_table)
    (hand-empty robot1)

    ;; All fruits clustered on main table (testing active perception)
    (fruit-at apple1 loc_main_table)
    (on apple1 main_table)
    (is-apple apple1)
    (is-red apple1)

    (fruit-at tomato1 loc_main_table)
    (on tomato1 main_table)
    (is-tomato tomato1)
    (is-red tomato1)

    (fruit-at strawberry1 loc_main_table)
    (on strawberry1 main_table)
    (is-strawberry strawberry1)
    (is-red strawberry1)

    (fruit-at orange1 loc_main_table)
    (on orange1 main_table)
    (is-orange orange1)
    (is-orange-colored orange1)

    ;; Surface relationships
    (is-surface main_table loc_main_table)
    (is-surface goal_table loc_goal_table)
    (is-goal-surface goal_table)
  )

  (:goal
    (and
      ;; Pick up the apple and tomato, place on goal table
      ;; This requires distinguishing between 3 similar red objects!
      (on apple1 goal_table)
      (on tomato1 goal_table)
      (fruit-at apple1 loc_goal_table)
      (fruit-at tomato1 loc_goal_table)

      ;; Robot should have empty hand at end
      (hand-empty robot1)
    )
  )
)
