(define (domain omnigibson)

    (:requirements :strips :typing :negative-preconditions :conditional-effects)

    (:types
        movable liquid furniture room agent - object

        wooden_stick tupperware brownie beer_bottle water_bottle mug pie carving_knife hard__boiled_egg - movable
        water - liquid
        countertop electric_refrigerator oven cabinet sink floor microwave table - furniture

        kitchen living_room - room

        water-n-06 - water
        mug-n-04 - mug
        cabinet-n-01 - cabinet
        sink-n-01 - sink
        floor-n-01 - floor
        microwave-n-02 - microwave
        pie-n-01 - pie
        oven-n-01 - oven
        electric_refrigerator-n-01 - electric_refrigerator
        carving_knife-n-01 - carving_knife
        countertop-n-01 - countertop
        hard__boiled_egg-n-01 - hard__boiled_egg
        water_bottle-n-01 - water_bottle
        beer_bottle-n-01 - beer_bottle
        brownie-n-03 - brownie
        tupperware-n-01 - tupperware
        wooden_stick-n-01 - wooden_stick
        table-n-02 - table

        agent-n-01 - agent
    )

    (:predicates
        (inside ?o1 - object ?o2 - object)
        (insource ?s - sink ?w - liquid)
        (inroom ?o - object ?r - room)
        (inhand ?a - agent ?o - object)
        (inview ?a - agent ?o - object)
        (handempty ?a - agent)
        (closed ?o - object)
        (filled ?o - movable ?w - liquid)
        (filledsink ?s - sink ?w - liquid)
        (turnedon ?o - object)
        (cooked ?o - object)
        (found ?a - agent ?o - object)
        (frozen ?o - object)
        (hot ?o - object)
        (halved ?o - object)
        (onfloor ?o - object ?f - floor)
        (ontop ?o1 - object ?o2 - object)
    )

    (:action find
        :parameters (?a - agent ?o - object ?r - room)
        :precondition (and (inroom ?a ?r) (inroom ?o ?r))
        :effect (and (inview ?a ?o) (found ?a ?o) (forall
                (?oo - object)
                (when
                    (found ?a ?oo)
                    (not (found ?a ?oo)))))
    )

    (:action grasp
        :parameters (?a - agent ?o1 - movable ?o2 - object)
        :precondition (and (inview ?a ?o1) (found ?a ?o1) (handempty ?a))
        :effect (and (not (inview ?a ?o1)) (not (handempty ?a)) (inhand ?a ?o1))
    )

    (:action placein
        :parameters (?a - agent ?o1 - movable ?o2 - object)
        :precondition (and (not (handempty ?a)) (inhand ?a ?o1) (inview ?a ?o2) (found ?a ?o2) (not (closed ?o2)))
        :effect (and (handempty ?a) (not (inhand ?a ?o1)) (inside ?o1 ?o2) (forall
                (?oo - object)
                (when
                    (inside ?oo ?o1)
                    (inside ?oo ?o2))
            ))
    )

    (:action placeon
        :parameters (?a - agent ?o1 - movable ?o2 - object)
        :precondition (and (not (handempty ?a)) (inhand ?a ?o1) (inview ?a ?o2) (found ?a ?o2))
        :effect (and (handempty ?a) (not (inhand ?a ?o1)) (ontop ?o1 ?o2))
    )

    (:action fillsink
        :parameters (?a - agent ?s - sink ?w - liquid)
        :precondition (and (inview ?a ?s) (found ?a ?s) (insource ?s ?w))
        :effect (filledsink ?s ?w)
    )

    (:action fill
        :parameters (?a - agent ?o - movable ?s - sink ?w - liquid)
        :precondition (and (inhand ?a ?o) (not (handempty ?a)) (filledsink ?s ?w) (inview ?a ?s) (found ?a ?s))
        :effect (and (filled ?o ?w) (not (filledsink ?s ?w)))
    )

    (:action openit
        :parameters (?a - agent ?o - object ?r - room)
        :precondition (and (inview ?a ?o) (found ?a ?o) (inroom ?o ?r))
        :effect (and (not (closed ?o)) (forall
                (?oo - object)
                (when
                    (inside ?oo ?o)
                    (inroom ?oo ?r))
            ))
    )

    (:action closeit
        :parameters (?a - agent ?o - object ?r - room)
        :precondition (and (inview ?a ?o) (found ?a ?o) (inroom ?o ?r))
        :effect (and (closed ?o) (forall
                (?oo - object)
                (when
                    (inside ?oo ?o)
                    (not (inroom ?oo ?r)))
                ; (and (not (inroom ?oo ?r)) (not (inview ?a ?oo))))
            ))
    )

    (:action microwave_water
        :parameters (?a - agent ?m - microwave ?o - movable ?w - water)
        :precondition (and (inview ?a ?m) (found ?a ?m) (closed ?m) (inside ?o ?m) (filled ?o ?w)) ; this will potentially cause problem, where the robot will stuck at checking if the microwave has the cup inside
        :effect (and (turnedon ?m) (cooked ?w))
    )

    (:action heat_food_with_oven
        :parameters (?a - agent ?v - oven ?f - object)
        :precondition (and (inview ?a ?v) (found ?a ?v) (inside ?f ?v))
        :effect (and (hot ?f) (turnedon ?v))
    )

    (:action cut_into_half
        :parameters (?a - agent ?k - carving_knife ?o - object)
        :precondition (and (inview ?a ?o) (found ?a ?o) (not (handempty ?a)) (inhand ?a ?k))
        :effect (halved ?o)
    )

    (:action place_on_floor
        :parameters (?a - agent ?o - object ?f - floor)
        :precondition (and (inview ?a ?f) (found ?a ?f) (not (handempty ?a)) (inhand ?a ?o))
        :effect (and (handempty ?a) (not (inhand ?a ?o)) (onfloor ?o ?f))
    )
)