;; PDDL Domain for Task 2: Halve an Egg
;; Use knife to cut hard-boiled egg into two halves

(define (domain egg_halving)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    robot
    item
    tool
    location
    room
    surface
  )

  (:predicates
    ;; Robot state
    (at ?r - robot ?loc - location)
    (holding ?r - robot ?i - item)
    (holding-tool ?r - robot ?t - tool)
    (hand-empty ?r - robot)

    ;; Item state
    (item-at ?i - item ?loc - location)
    (on ?i - item ?s - surface)
    (is-egg ?i - item)
    (is-whole ?i - item)
    (is-halved ?i - item)

    ;; Tool state
    (tool-at ?t - tool ?loc - location)
    (on-surface ?t - tool ?s - surface)
    (is-knife ?t - tool)
    (is-sharp ?t - tool)
    (tool-dropped ?t - tool)  ;; Situation: tool fell during operation

    ;; Location relationships
    (in-room ?loc - location ?rm - room)
    (is-surface ?s - surface ?loc - location)
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

  (:action pick-tool
    :parameters (?r - robot ?t - tool ?loc - location ?s - surface)
    :precondition (and
      (at ?r ?loc)
      (tool-at ?t ?loc)
      (on-surface ?t ?s)
      (hand-empty ?r)
    )
    :effect (and
      (holding-tool ?r ?t)
      (not (hand-empty ?r))
      (not (tool-at ?t ?loc))
      (not (on-surface ?t ?s))
    )
  )

  (:action place-tool
    :parameters (?r - robot ?t - tool ?loc - location ?s - surface)
    :precondition (and
      (at ?r ?loc)
      (holding-tool ?r ?t)
      (is-surface ?s ?loc)
    )
    :effect (and
      (not (holding-tool ?r ?t))
      (hand-empty ?r)
      (tool-at ?t ?loc)
      (on-surface ?t ?s)
    )
  )

  (:action cut-egg
    :parameters (?r - robot ?egg - item ?knife - tool ?loc - location ?s - surface)
    :precondition (and
      (at ?r ?loc)
      (holding-tool ?r ?knife)
      (is-knife ?knife)
      (is-sharp ?knife)
      (item-at ?egg ?loc)
      (on ?egg ?s)
      (is-egg ?egg)
      (is-whole ?egg)
    )
    :effect (and
      (not (is-whole ?egg))
      (is-halved ?egg)
    )
  )

  ;; Recovery action: pick up a dropped tool
  (:action pick-dropped-tool
    :parameters (?r - robot ?t - tool ?loc - location ?s - surface)
    :precondition (and
      (at ?r ?loc)
      (tool-at ?t ?loc)
      (on-surface ?t ?s)
      (hand-empty ?r)
      (tool-dropped ?t)
    )
    :effect (and
      (not (tool-dropped ?t))
      (holding-tool ?r ?t)
      (not (hand-empty ?r))
      (not (tool-at ?t ?loc))
      (not (on-surface ?t ?s))
    )
  )
)
