(define (problem store_wooden_stick-0)
    (:domain omnigibson)

    (:objects
        wooden_stick-n-01_2 wooden_stick-n-01_3 - wooden_stick-n-01
        floor-n-01_1 - floor-n-01
        table-n-02_1 - table-n-02
        agent-n-01_1 - agent-n-01
        living_room - room
    )

    (:init
        (inroom wooden_stick-n-01_2 living_room)
        (inroom wooden_stick-n-01_3 living_room)
        (inroom floor-n-01_1 living_room)
        (inroom table-n-02_1 living_room)
        (inroom agent-n-01_1 living_room)
        (handempty agent-n-01_1)
    )

    (:goal
        (and
            (ontop wooden_stick-n-01_2 table-n-02_1)
            (ontop wooden_stick-n-01_3 table-n-02_1)
        )
    )
)