(define (problem boil_water_in_the_microwave-0)
    (:domain omnigibson)

    (:objects
        water-n-06_1 - water-n-06
        mug-n-04_1 - mug-n-04
        cabinet-n-01_1 - cabinet-n-01
        sink-n-01_1 - sink-n-01
        floor-n-01_1 - floor-n-01
        microwave-n-02_1 - microwave-n-02
        agent-n-01_1 - agent-n-01
        kitchen - room
    )

    (:init
        (closed cabinet-n-01_1)
        (closed microwave-n-02_1)
        (inside mug-n-04_1 cabinet-n-01_1)
        (insource sink-n-01_1 water-n-06_1)
        (inroom floor-n-01_1 kitchen)
        (inroom sink-n-01_1 kitchen)
        (inroom microwave-n-02_1 kitchen)
        (inroom cabinet-n-01_1 kitchen)
        (handempty agent-n-01_1)
        (inroom agent-n-01_1 kitchen)
    )

    (:goal
        (and
            (cooked water-n-06_1)
        )
    )
)