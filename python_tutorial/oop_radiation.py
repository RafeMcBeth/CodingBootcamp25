"""Example of object-oriented programming for simple radiation calculations."""

import math


class RadiationSource:
    def __init__(self, activity_bq, half_life_hours):
        self.activity_bq = activity_bq
        self.half_life_hours = half_life_hours

    def decayed_activity(self, time_hours):
        """Calculate remaining activity after ``time_hours``."""
        decay_constant = math.log(2) / self.half_life_hours
        return self.activity_bq * math.exp(-decay_constant * time_hours)


class ExternalBeam(RadiationSource):
    def __init__(self, activity_bq, half_life_hours, beam_energy_mev):
        super().__init__(activity_bq, half_life_hours)
        self.beam_energy_mev = beam_energy_mev

    def info(self):
        return f"External beam {self.beam_energy_mev} MeV, initial activity {self.activity_bq} Bq"


def main():
    cobalt_source = RadiationSource(activity_bq=3.7e10, half_life_hours=1925)
    beam = ExternalBeam(activity_bq=1e9, half_life_hours=24, beam_energy_mev=6)

    print(cobalt_source.decayed_activity(48))
    print(beam.info())


if __name__ == "__main__":
    main()
