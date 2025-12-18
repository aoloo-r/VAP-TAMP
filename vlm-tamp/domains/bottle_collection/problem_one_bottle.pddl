;; PDDL Problem for Task 1: Bring One Empty Bottle (Test)
;; Collect only bottle1 from room2 and place in collection area in room1

(define (problem collect_one_bottle)
  (:domain bottle_collection)

  (:objects
    robot1 - robot

    ;; Items
    bottle1 - item

    ;; Rooms
    room1 room2 - room  ; room1 = main room (lab)

    ;; Locations
    loc_main loc_room2 - location

    ;; Surfaces
    table_room2 collection_table - surface
  )

  (:init
    ;; Robot starts in main room
    (at robot1 loc_main)
    (hand-empty robot1)

    ;; Bottle 1 in room2 (TA Area)
    (item-at bottle1 loc_room2)
    (on bottle1 table_room2)
    (is-bottle bottle1)
    (is-empty bottle1)

    ;; Location-room relationships
    (in-room loc_main room1)
    (in-room loc_room2 room2)

    ;; Surface-location relationships
    (is-surface table_room2 loc_room2)
    (is-surface collection_table loc_main)

    ;; Collection area designation
    (is-collection-area collection_table)

    ;; Room connectivity (bidirectional)
    (connected room1 room2)
    (connected room2 room1)
  )

  (:goal
    (and
      ;; Bottle in collection area
      (on bottle1 collection_table)
      (item-at bottle1 loc_main)
    )
  )
)
