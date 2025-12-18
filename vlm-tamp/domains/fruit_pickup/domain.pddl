;; PDDL Domain for Active Perception Testing: Fruit Pickup
;; Test scenario with similar-colored fruits to necessitate active perception
;;
;; Challenge: Multiple red/similar colored fruits on table:
;;   - Red apple
;;   - Red tomato
;;   - Strawberry
;;   - Orange (for contrast)
;;
;; Task: Pick up specific fruits from the table
;; This requires the robot to distinguish between similar-looking objects

(define (domain fruit_pickup)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    robot
    fruit
    location
    surface
  )

  (:predicates
    ;; Robot state
    (at ?r - robot ?loc - location)
    (holding ?r - robot ?f - fruit)
    (hand-empty ?r - robot)

    ;; Fruit state
    (fruit-at ?f - fruit ?loc - location)
    (on ?f - fruit ?s - surface)

    ;; Fruit types (for VLM verification - these are challenging!)
    (is-apple ?f - fruit)
    (is-tomato ?f - fruit)
    (is-strawberry ?f - fruit)
    (is-orange ?f - fruit)

    ;; Color predicates (multiple fruits share same color - requires active perception)
    (is-red ?f - fruit)
    (is-orange-colored ?f - fruit)

    ;; Location relationships
    (is-surface ?s - surface ?loc - location)
    (is-goal-surface ?s - surface)
  )

  (:action navigate
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (at ?r ?from)
    :effect (and
      (not (at ?r ?from))
      (at ?r ?to)
    )
  )

  (:action pick
    :parameters (?r - robot ?f - fruit ?loc - location ?s - surface)
    :precondition (and
      (at ?r ?loc)
      (fruit-at ?f ?loc)
      (on ?f ?s)
      (hand-empty ?r)
    )
    :effect (and
      (holding ?r ?f)
      (not (hand-empty ?r))
      (not (fruit-at ?f ?loc))
      (not (on ?f ?s))
    )
  )

  (:action place
    :parameters (?r - robot ?f - fruit ?loc - location ?s - surface)
    :precondition (and
      (at ?r ?loc)
      (holding ?r ?f)
      (is-surface ?s ?loc)
    )
    :effect (and
      (not (holding ?r ?f))
      (hand-empty ?r)
      (fruit-at ?f ?loc)
      (on ?f ?s)
    )
  )
)
