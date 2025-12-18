(define (problem bringing_water-0)
    (:domain omnigibson)

    (:objects
        water_bottle-n-01_1 water_bottle-n-01_2 - water_bottle-n-01
        floor-n-01_1 floor-n-01_2 - floor-n-01
        agent-n-01_1 - agent-n-01
        kitchen - room
    )

    (:init
        (inroom water_bottle-n-01_1 kitchen)
        (inroom water_bottle-n-01_2 kitchen)
        (inroom floor-n-01_1 kitchen)
        (inroom floor-n-01_2 kitchen)
        (inroom agent-n-01_1 kitchen)
        (handempty agent-n-01_1)
    )

    (:goal
        (and
            (onfloor water_bottle-n-01_1 floor-n-01_1)
            (onfloor water_bottle-n-01_2 floor-n-01_1)
        )
    )
)