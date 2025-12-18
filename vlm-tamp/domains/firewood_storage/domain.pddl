;; PDDL Domain for Task 3: Store Firewood
;; Collect two wooden sticks from another room, handle closed door

(define (domain firewood_storage)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    robot
    item
    location
    room
    surface
    door
  )

  (:predicates
    ;; Robot state
    (at ?r - robot ?loc - location)
    (holding ?r - robot ?i - item)
    (hand-empty ?r - robot)

    ;; Item state
    (item-at ?i - item ?loc - location)
    (on ?i - item ?s - surface)
    (is-wood ?i - item)
    (is-stick ?i - item)

    ;; Door state (requires active perception to distinguish)
    (door-between ?d - door ?rm1 - room ?rm2 - room)
    (door-fully-open ?d - door)
    (door-half-open ?d - door)
    (door-closed ?d - door)
    (door-access-requested ?d - door)

    ;; Location relationships
    (in-room ?loc - location ?rm - room)
    (is-surface ?s - surface ?loc - location)
    (is-storage-table ?s - surface)

    ;; Room connectivity
    (connected ?rm1 - room ?rm2 - room)
  )

  (:action navigate
    :parameters (?r - robot ?from - location ?to - location ?rm - room)
    :precondition (and
      (at ?r ?from)
      (in-room ?to ?rm)
    )
    :effect (and
      (not (at ?r ?from))
      (at ?r ?to)
    )
  )

  (:action request-door-open
    :parameters (?r - robot ?d - door ?rm1 - room ?rm2 - room ?loc - location)
    :precondition (and
      (at ?r ?loc)
      (in-room ?loc ?rm1)
      (door-between ?d ?rm1 ?rm2)
      (door-closed ?d)
    )
    :effect (and
      (door-access-requested ?d)
    )
  )

  (:action request-door-open-more
    :parameters (?r - robot ?d - door ?rm1 - room ?rm2 - room ?loc - location)
    :precondition (and
      (at ?r ?loc)
      (in-room ?loc ?rm1)
      (door-between ?d ?rm1 ?rm2)
      (door-half-open ?d)
    )
    :effect (and
      (door-access-requested ?d)
    )
  )

  (:action pass-through-door
    :parameters (?r - robot ?from - location ?to - location ?d - door ?rm1 - room ?rm2 - room)
    :precondition (and
      (at ?r ?from)
      (in-room ?from ?rm1)
      (in-room ?to ?rm2)
      (door-between ?d ?rm1 ?rm2)
      (door-fully-open ?d)
    )
    :effect (and
      (not (at ?r ?from))
      (at ?r ?to)
    )
  )

  (:action pick
    :parameters (?r - robot ?i - item ?loc - location ?s - surface)
    :precondition (and
      (at ?r ?loc)
      (item-at ?i ?loc)
      (on ?i ?s)
      (hand-empty ?r)
    )
    :effect (and
      (holding ?r ?i)
      (not (hand-empty ?r))
      (not (item-at ?i ?loc))
      (not (on ?i ?s))
    )
  )

  (:action place
    :parameters (?r - robot ?i - item ?loc - location ?s - surface)
    :precondition (and
      (at ?r ?loc)
      (holding ?r ?i)
      (is-surface ?s ?loc)
    )
    :effect (and
      (not (holding ?r ?i))
      (hand-empty ?r)
      (item-at ?i ?loc)
      (on ?i ?s)
    )
  )
)
