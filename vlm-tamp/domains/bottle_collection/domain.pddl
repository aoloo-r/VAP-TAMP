;; PDDL Domain for Task 1: Bring Empty Bottles
;; Two empty bottles in different rooms -> designated area in main room

(define (domain bottle_collection)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    robot
    item
    location
    room
    surface
  )

  (:predicates
    ;; Robot state
    (at ?r - robot ?loc - location)
    (holding ?r - robot ?i - item)
    (hand-empty ?r - robot)

    ;; Item state
    (item-at ?i - item ?loc - location)
    (on ?i - item ?s - surface)
    (is-bottle ?i - item)
    (is-empty ?i - item)
    (grasp-failed ?i - item)  ;; Situation: grasp was unsuccessful

    ;; Location relationships
    (in-room ?loc - location ?rm - room)
    (is-surface ?s - surface ?loc - location)
    (is-collection-area ?s - surface)

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

  ;; Recovery action: re-grasp an object that failed to grasp
  (:action regrasping
    :parameters (?r - robot ?i - item ?loc - location ?s - surface)
    :precondition (and
      (at ?r ?loc)
      (item-at ?i ?loc)
      (on ?i ?s)
      (hand-empty ?r)
      (grasp-failed ?i)
    )
    :effect (and
      (not (grasp-failed ?i))
      (holding ?r ?i)
      (not (hand-empty ?r))
      (not (item-at ?i ?loc))
      (not (on ?i ?s))
    )
  )
)
