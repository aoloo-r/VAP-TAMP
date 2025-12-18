;; PDDL Problem for Task 3: Store Firewood
;; Collect two wooden sticks from room 3, return to room 1 through room 2
;; Door between Room 1 and Room 2 is closed - requires human assistance

(define (problem store_firewood)
  (:domain firewood_storage)

  (:objects
    robot1 - robot

    ;; Items
    stick1 stick2 - item

    ;; Rooms
    room1 room2 room3 - room

    ;; Locations
    loc_main loc_room2 loc_room3 - location

    ;; Surfaces
    floor_room3 main_table - surface

    ;; Door (between Room 1 and Room 2)
    door_room1_to_room2 - door
  )

  (:init
    ;; Robot starts in room 2 (same as bottle collection location in Task 1)
    (at robot1 loc_room2)
    (hand-empty robot1)

    ;; Sticks in room 3
    (item-at stick1 loc_room3)
    (on stick1 floor_room3)
    (is-wood stick1)
    (is-stick stick1)

    (item-at stick2 loc_room3)
    (on stick2 floor_room3)
    (is-wood stick2)
    (is-stick stick2)

    ;; Door between Room 1 and Room 2 starts half-open
    ;; SITUATION & ACTIVE PERCEPTION: Door is half-open, requires multiple viewpoints to verify state
    ;; - Robot must use active perception to distinguish half-open vs fully-open
    ;; - If half-open, request human to open fully before passing through
    ;; Path: Room 2 (START) -> Room 3 (collect sticks) -> Room 2 -> (CHECK DOOR) -> Room 1 (deliver)
    (door-between door_room1_to_room2 room1 room2)
    (door-half-open door_room1_to_room2)
    (not (door-fully-open door_room1_to_room2))
    (not (door-closed door_room1_to_room2))

    ;; Location-room relationships
    (in-room loc_main room1)
    (in-room loc_room2 room2)
    (in-room loc_room3 room3)

    ;; Surface-location relationships
    (is-surface floor_room3 loc_room3)
    (is-surface main_table loc_main)

    ;; Storage table designation
    (is-storage-table main_table)

    ;; Room connectivity: Room 3 -> Room 2 -> Room 1
    (connected room1 room2)
    (connected room2 room1)
    (connected room2 room3)
    (connected room3 room2)
  )

  (:goal
    (and
      ;; Both sticks on storage table in main room
      (on stick1 main_table)
      (on stick2 main_table)
      (item-at stick1 loc_main)
      (item-at stick2 loc_main)
    )
  )
)
