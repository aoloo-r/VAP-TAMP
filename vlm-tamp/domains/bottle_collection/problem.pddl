;; PDDL Problem for Task 1: Bring Empty Bottles
;; Collect bottle1 from room2 and bottle2 from room3, place in collection area in room1

(define (problem collect_bottles)
  (:domain bottle_collection)

  (:objects
    robot1 - robot

    ;; Items
    bottle1 bottle2 - item

    ;; Rooms
    room1 room2 room3 - room  ; room1 = main room

    ;; Locations
    loc_main loc_room2 loc_room3 - location

    ;; Surfaces
    table_room2 table_room3 collection_table - surface
  )

  (:init
    ;; Robot starts in main room
    (at robot1 loc_main)
    (hand-empty robot1)

    ;; Bottle 1 in room2
    (item-at bottle1 loc_room2)
    (on bottle1 table_room2)
    (is-bottle bottle1)
    (is-empty bottle1)

    ;; Bottle 2 in room3
    (item-at bottle2 loc_room3)
    (on bottle2 table_room3)
    (is-bottle bottle2)
    (is-empty bottle2)

    ;; Location-room relationships
    (in-room loc_main room1)
    (in-room loc_room2 room2)
    (in-room loc_room3 room3)

    ;; Surface-location relationships
    (is-surface table_room2 loc_room2)
    (is-surface table_room3 loc_room3)
    (is-surface collection_table loc_main)

    ;; Collection area designation
    (is-collection-area collection_table)

    ;; Room connectivity (bidirectional)
    (connected room1 room2)
    (connected room2 room1)
    (connected room1 room3)
    (connected room3 room1)
    (connected room2 room3)
    (connected room3 room2)
  )

  (:goal
    (and
      ;; Both bottles in collection area
      (on bottle1 collection_table)
      (on bottle2 collection_table)
      (item-at bottle1 loc_main)
      (item-at bottle2 loc_main)
    )
  )
)
