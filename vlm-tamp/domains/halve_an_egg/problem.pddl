(define (problem halve_an_egg-0)
    (:domain omnigibson)

    (:objects
        carving_knife-n-01_1 - carving_knife-n-01
        countertop-n-01_1 - countertop-n-01
        hard__boiled_egg-n-01_1 - hard__boiled_egg-n-01
        floor-n-01_1 - floor-n-01
        agent-n-01_1 - agent-n-01
        kitchen - room
    )

    (:init
        (inroom countertop-n-01_1 kitchen)
        (inroom carving_knife-n-01_1 kitchen)
        (inroom floor-n-01_1 kitchen)
        (inroom hard__boiled_egg-n-01_1 kitchen)
        (handempty agent-n-01_1)
        (inroom agent-n-01_1 kitchen)
    )

    (:goal
        (halved hard__boiled_egg-n-01_1)
    )
)