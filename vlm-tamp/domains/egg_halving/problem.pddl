;; PDDL Problem for Task 2: Halve an Egg
;; Find knife and use it to cut hard-boiled egg in Room 1 (Lab)
;; Egg location: same as Task 1 start position (-13.0133, -15.1409)

(define (problem halve_egg)
  (:domain egg_halving)

  (:objects
    robot1 - robot

    ;; Items
    egg1 - item

    ;; Tools
    knife1 - tool

    ;; Rooms
    room1 - room

    ;; Locations
    loc_main - location  ;; Room 1 Lab location (egg)
    loc_knife - location  ;; Knife location

    ;; Surfaces
    counter kitchen_table - surface
  )

  (:init
    ;; Robot starts at Room 1 (Lab) location
    (at robot1 loc_main)
    (hand-empty robot1)

    ;; Egg on kitchen table in Room 1 (Lab)
    ;; AMCL location: (-13.0133, -15.1409)
    (item-at egg1 loc_main)
    (on egg1 kitchen_table)
    (is-egg egg1)
    (is-whole egg1)

    ;; Knife on counter
    ;; AMCL location: (-7.06018585, -20.45418739)
    (tool-at knife1 loc_knife)
    (on-surface knife1 counter)
    (is-knife knife1)
    (is-sharp knife1)

    ;; Location-room relationships
    (in-room loc_main room1)
    (in-room loc_knife room1)

    ;; Surface-location relationships
    (is-surface counter loc_knife)
    (is-surface kitchen_table loc_main)
  )

  (:goal
    (and
      ;; Egg is halved
      (is-halved egg1)
      (not (is-whole egg1))
    )
  )
)
