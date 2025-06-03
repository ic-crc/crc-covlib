# Copyright (c) 2025 His Majesty the King in Right of Canada as represented by the Minister of
# Industry through the Communications Research Centre Canada.
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license text.

import sys, os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../../'))
from crc_covlib import simulation as covlib
import ray
import time


# NOTE: It is safe to call methods of different Simulation objects in parallel.
#       However do not call methods of a same Simulation object in parallel.
@ray.remote
def run_sim_on_ray(sim_name: str, freq_MHz: float, output_pathname: str) -> str:
    run_sim(freq_MHz, output_pathname)
    return sim_name


def run_sim(freq_MHz: float, output_pathname: str) -> None:
    sim = covlib.Simulation()

    # Set transmitter parameters
    sim.SetTransmitterLocation(45.42531, -75.71573)
    sim.SetTransmitterHeight(30)
    sim.SetTransmitterPower(2, covlib.PowerType.EIRP)
    sim.SetTransmitterFrequency(freq_MHz)

    # Set receiver parameters
    sim.SetReceiverHeightAboveGround(1.5)

    # Propagation model selection
    P1812 = covlib.PropagationModel.ITU_R_P_1812
    sim.SetPropagationModel(P1812)

    # Specify file to get ITU radio climate zones from.
    sim.SetITURP1812RadioClimaticZonesFile(os.path.join(script_dir, '../../../data/itu-radio-climatic-zones/rcz.tif'))

    # Set terrain elevation data parameters
    CDEM = covlib.TerrainElevDataSource.TERR_ELEV_NRCAN_CDEM
    sim.SetPrimaryTerrainElevDataSource(CDEM)
    sim.SetTerrainElevDataSourceDirectory(CDEM, os.path.join(script_dir, '../../../data/terrain-elev-samples/NRCAN_CDEM'))
    sim.SetTerrainElevDataSamplingResolution(25)

    # Set land cover data parameters
    WORLDCOVER = covlib.LandCoverDataSource.LAND_COVER_ESA_WORLDCOVER
    sim.SetPrimaryLandCoverDataSource(WORLDCOVER)
    sim.SetLandCoverDataSourceDirectory(WORLDCOVER, os.path.join(script_dir, '../../../data/land-cover-samples/ESA_Worldcover'))

    # Set reception/coverage area parameters
    sim.SetReceptionAreaCorners(45.37914, -75.81922, 45.47148, -75.61225)
    sim.SetReceptionAreaNumHorizontalPoints(200)
    sim.SetReceptionAreaNumVerticalPoints(200)
    sim.SetResultType(covlib.ResultType.FIELD_STRENGTH_DBUVM)

    sim.GenerateReceptionAreaResults()
    sim.ExportReceptionAreaResultsToBilFile(output_pathname)


if __name__ == '__main__':

    print('\ncrc-covlib - Sequential simulations VS local parallel simulations using Ray\n')

    frequencies_MHz = [1500, 2000, 2500, 3000]

    # Run 4 simulations at different frequencies sequentially
    start_time = time.time()
    for i, f_MHz in enumerate(frequencies_MHz):
        output_file = os.path.join(script_dir, 'sim_{}.bil'.format(i+1))
        run_sim(f_MHz, output_file)
        print('sim_{} completed (sequential)'.format(i+1))
    end_time = time.time()
    print('elpased time (sequential): {:.3f} secs\n'.format(end_time-start_time))


    # Run simulations in parallel

    start_time_with_init = time.time()

    # Change the current working directory to the folder containing the crc_covlib package, otherwise Ray
    # may not find it (works when running Ray locally, may not help otherwise).
    os.chdir(os.path.join(script_dir, '../../'))
    os.getcwd() # ensure the working directory change has completed before ray.init() is called (fix an occasional issue noticed on Ubuntu)
    ray.init()

    # Alternately, we could do the following:
    """
    # This will copy the content of the specified local working_dir to the cluster nodes' runtime environment
    # directory (here no remote cluster is specified so files be will zipped/copied/unzipped to a temporary local
    # folder). We exclude some files and folders so that only the crc_covlib package folder will be copied.
    # For more details, see https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference
    #                   and https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-init
    #                   and https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#environment-dependencies
    runtime_env = {'working_dir': os.path.join(script_dir, '../../'),
                   'excludes': ['/tests/', '/docs/', '/examples/', '/pyproject.toml', '/README.md', '/README_FR.md',
                                '/crc_covlib/helper/'] } # helper modules may be excluded from copy when not used
    # This runs on a local machine, but you can also do ray.init(address=..., runtime_env=...) to connect to a cluster.
    ray.init(runtime_env=runtime_env)
    """

    start_time_no_init = time.time()

    ids = []
    for i, f_MHz in enumerate(frequencies_MHz):
        sim_name = 'sim_{}'.format(i+1)
        output_file = os.path.join(script_dir, 'ray_sim_{}.bil'.format(i+1))
        ids.append( run_sim_on_ray.remote(sim_name, f_MHz, output_file))

    while True:
        [ready_id], not_ready_ids = ray.wait(ids, num_returns=1)
        sim_name = ray.get(ready_id)
        print('{} completed (parallel)'.format(sim_name))
        ids = not_ready_ids
        if not ids:
            break

    end_time = time.time()
    print('elpased time including ray.init (parallel): {:.3f} secs'.format(end_time-start_time_with_init))
    print('elpased time excluding ray.init (parallel): {:.3f} secs'.format(end_time-start_time_no_init))

    print('\nSimulations completed\n')
