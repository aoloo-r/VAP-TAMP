(define (problem cook_a_frozen_pie-0)
    (:domain omnigibson)

    (:objects
        pie-n-01_1 - pie-n-01
        oven-n-01_1 - oven-n-01
        electric_refrigerator-n-01_1 - electric_refrigerator-n-01
        floor-n-01_1 - floor-n-01
        agent-n-01_1 - agent-n-01
        kitchen - room
    )

    (:init
        (closed oven-n-01_1)
        (closed electric_refrigerator-n-01_1)
        (inside pie-n-01_1 electric_refrigerator-n-01_1)
        (frozen pie-n-01_1)
        (inroom oven-n-01_1 kitchen)
        (inroom floor-n-01_1 kitchen)
        (inroom electric_refrigerator-n-01_1 kitchen)
        (inroom agent-n-01_1 kitchen)
        (handempty agent-n-01_1)
    )

    (:goal
        (and
            (hot pie-n-01_1)
        )
    )
)