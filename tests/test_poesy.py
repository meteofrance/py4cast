from py4cast.datasets.poesy import Period, PoesySettings, run_through_timestamps


def test_run_through_timestamps():
    """
    Test the browse_dataset function
    """
    # Test period odd 
    period = Period(start="2021061500", end="2021061503", step=2, name="test")

    term = {"start": 0, "end": 7, "timestep": 2}

    settings = PoesySettings(
        members=[0, 1],
        term=term,
        num_output_steps=1,
        num_input_steps=1,
    )

    list_args_all_samples = run_through_timestamps(period, settings)

    # Test time * term * member
    assert len(list_args_all_samples) == 2 * 2 * 2
    
    # Test period even 
    period = Period(start="2021061500", end="2021061504", step=2, name="test")
    list_args_all_samples = run_through_timestamps(period, settings)

    # Test time * term * member
    assert len(list_args_all_samples) == 2 * 2 * 2

if __name__ == "__main__":
    test_run_through_timestamps()
