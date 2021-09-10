# PySpice
import unittest
import PySpice.Unit
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
import PySpice.Logging.Logging as Logging
import pint
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from PySpice.Physics.SemiConductor import ShockleyDiode


class UnitTestsElectronicSimulationPySpiceExamples(unittest.TestCase):
    def test_internal_device_parameters(self):
        print("test_internal_device_parameters")

        logger = Logging.setup_logging()

        class Level1(SubCircuitFactory):
            NAME = 'level1'
            NODES = ('d3', 'g3', 'v3')

            def __init__(self):
                super().__init__()
                self.X('mos2', 'level2', 'd3', 'g3', 'v3')
                self.subcircuit(Level2())

        class Level2(SubCircuitFactory):
            NAME = 'level2'
            NODES = ('d4', 'g4', 'v4')

            def __init__(self):
                super().__init__()
                self.M(1, 'd4', 'g4', 'v4', 'v4', model='NMOS', w=1e-5, l=3.5e-7)

        circuit = Circuit('Transistor output characteristics')
        circuit.V('dd', 'd1', circuit.gnd, 2)
        circuit.V('ss', 'vsss', circuit.gnd, 0)
        circuit.V('sig', 'g1', 'vsss', 0)
        circuit.X('mos1', 'level1', 'd1', 'g1', 'vsss')

        if True:
            circuit.subcircuit(Level1())
        else:
            subcircuit_level1 = SubCircuit('level1', 'd3', 'g3', 'v3')
            subcircuit_level1.X('mos2', 'level2', 'd3', 'g3', 'v3')
            subcircuit_level1.subcircuit(subcircuit_level2)

            subcircuit_level2 = SubCircuit('level2', 'd4', 'g4', 'v4')
            subcircuit_level2.M(1, 'd4', 'g4', 'v4', 'v4', model='NMOS', w=1e-5, l=3.5e-7)

            circuit.subcircuit(subcircuit_level1)

        circuit.model('NMOS', 'NMOS', LEVEL=8)

        print(str(circuit))

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        # Fixme: python return code is not 0 on Windows if the following line is executed
        #        but any error is reported
        # analysis = simulator.dc(Vdd=slice(0, 5, .1)) # Fixme: ,Vsig=slice(1, 5, 1)

        # To be completed …

    def test_netlist_manipulations(self):
        print("test_netlist_manipulations")

        logger = Logging.setup_logging()

        class SubCircuit1(SubCircuitFactory):
            NAME = 'sub_circuit1'
            NODES = ('n1', 'n2')

            def __init__(self):
                super().__init__()
                self.R(1, 'n1', 'n2', 1)
                self.R(2, 'n1', 'n2', 2)

        circuit = Circuit('Test')

        C1 = circuit.C(1, 0, 1, 1)

        circuit.C(2, 1, 2, 2)
        circuit.subcircuit(SubCircuit1())
        circuit.X('1', 'sub_circuit1', 2, 0)

        C1 = circuit.C1
        C1 = circuit['C1']

        C1.capacitance = 10

        # str(circuit) is implicit here
        print(str(circuit))

        print(C1)

        C1.enabled = False
        print(circuit)

        circuit2 = circuit.clone(title='A clone')  # title is optional
        print(circuit2)

        C2 = circuit2.C2.detach()
        print(circuit2)

    def test_pass_raw_spice_definitions_to_a_netlist(self):
        print("test_pass_raw_spice_definitions_to_a_netlist")

        logger = Logging.setup_logging()

        circuit = Circuit('Test')

        circuit.raw_spice = '''
        Vinput in 0 10V
        R1 in out 9kOhm
        '''

        circuit.R(2, 'out', 0, raw_spice='1k')

        print(circuit)

    def test_how_to_use_subcircuit(self):
        print("test_how_to_use_subcircuit")

        logger = Logging.setup_logging()

        class ParallelResistor(SubCircuitFactory):
            NAME = 'parallel_resistor'
            NODES = ('n1', 'n2')

            def __init__(self, R1=1, R2=2):
                super().__init__()
                self.R(1, 'n1', 'n2', R1)
                self.R(2, 'n1', 'n2', R2)

        circuit = Circuit('Test')

        circuit.subcircuit(ParallelResistor(R2=3))
        circuit.X('1', 'parallel_resistor', 1, circuit.gnd)

        print(circuit)

        class ParallelResistor2(SubCircuit):
            NODES = ('n1', 'n2')

            def __init__(self, name, R1=1, R2=2):
                SubCircuit.__init__(self, name, *self.NODES)
                self.R(1, 'n1', 'n2', R1)
                self.R(2, 'n1', 'n2', R2)

        circuit = Circuit('Test')
        circuit.subcircuit(ParallelResistor2('pr1', R2=2))
        circuit.X('1', 'pr1', 1, circuit.gnd)
        circuit.subcircuit(ParallelResistor2('pr2', R2=3))
        circuit.X('2', 'pr2', 1, circuit.gnd)

        print(circuit)

    def test_unit(self):
        print("test_unit")

        logger = Logging.setup_logging()

        foo = 1*10^(3)  # unit less

        resistance_unit = PySpice.Unit.unit.U_Ω

        resistance1 = PySpice.Unit.Unit.Unit.u_kΩ(1)
        resistance1 = PySpice.Unit.Unit.Unit.u_kOhm(1)  # ASCII variant

        resistance1 = PySpice.Unit.SiUnits.Ohm # using Python 3.5 syntax
        # resistance1 = 1@u_kΩ  # space doesn't matter
        # resistance1 = 1 @ u_kΩ  #

        # resistance2 = as_Ω(resistance1)  # check unit

        # resistances = u_kΩ(range(1, 11))  # same as [u_kΩ(x) for x in range(1, 11)]
        # resistances = range(1, 11) @ u_kΩ  # using Python 3.5 syntax

        # capacitance = u_uF(200)
        # inductance = u_mH(1)
        # temperature = u_Degree(25)

        # voltage = resistance1 * u_mA(1)  # compute unit

        # frequency = u_ms(20).frequency
        period = PySpice.Unit.FrequencyValue(50)
        # pulsation = frequency.pulsation
        pulsation = period.pulsation

        circuit = Circuit('Resistor Bridge')

        # resistance = 10 @ u_kΩ
        # print(float(resistance))
        # print(str(resistance))

        # circuit.V('input', 1, circuit.gnd, 10 @ u_V)
        # circuit.R(1, 1, 2, 2 @ u_kΩ)
        # circuit.R(2, 1, 3, 1 @ u_kΩ)
        # circuit.R(3, 2, circuit.gnd, 1 @ u_kΩ)
        # circuit.R(4, 3, circuit.gnd, 2 @ u_kΩ)
        # circuit.R(5, 3, 2, 2 @ u_kΩ)

        print(circuit)

        u = pint.UnitRegistry()

        resistance = 10 * u.kΩ
        # print(float(resistance))
        print(resistance.magnitude)
        print(resistance.m)
        print(resistance.units)
        print(str(resistance))

        circuit = Circuit('Resistor Bridge')

        circuit.V('input', 1, circuit.gnd, 10 * u.V)
        circuit.R(1, 1, 2, 2 * u.kΩ)
        circuit.R(2, 1, 3, 1 * u.kΩ)
        circuit.R(3, 2, circuit.gnd, 1 * u.kΩ)
        circuit.R(4, 3, circuit.gnd, 2 * u.kΩ)
        circuit.R(5, 3, 2, 2 * u.kΩ)

        print(circuit)

    def test_fast_fourier_transform(self):
        print("test_fast_fourier_transform")

        N = 1000  # number of sample points
        dt = 1. / 500  # sample spacing

        frequency1 = 50.
        frequency2 = 80.

        t = np.linspace(0.0, N * dt, N)
        y = np.sin(2 * np.pi * frequency1 * t) + .5 * np.sin(2 * np.pi * frequency2 * t)

        yf = fft(y)
        tf = np.linspace(.0, 1. / (2. * dt), N // 2)
        spectrum = 2. / N * np.abs(yf[0:N // 2])

        figure1, ax = plt.subplots(figsize=(20, 10))

        ax.plot(tf, spectrum, 'o-')

        ax.grid()

        for frequency in frequency1, frequency2:
            ax.axvline(x=frequency, color='red')

        ax.set_title('Spectrum')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')

        N = 1000  # number of sample points
        dt = 1. / 1000  # sample spacing

        frequency = 5.

        t = np.linspace(.0, N * dt, N)
        y = signal.square(2 * np.pi * frequency * t)

        figure2, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

        ax1.plot(t, y)

        y_sum = None

        for n in range(1, 20, 2):
            yn = 4 / (np.pi * n) * np.sin((2 * np.pi * n * frequency * t))
            if y_sum is None:
                y_sum = yn
            else:
                y_sum += yn
            if n in (1, 3, 5):
                ax1.plot(t, y_sum)

        ax1.plot(t, y_sum)
        ax1.set_xlim(0, 2 / frequency)
        ax1.set_ylim(-1.5, 1.5)

        yf = fft(y)
        tf = np.linspace(.0, 1. / (2. * dt), N // 2)
        spectrum = 2. / N * np.abs(yf[0:N // 2])

        ax2.plot(tf, spectrum)
        n = np.arange(1, 20, 2)
        ax2.plot(n * frequency, 4 / (np.pi * n), 'o', color='red')
        ax2.grid()
        ax2.set_title('Spectrum')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Amplitude')

        plt.show()

    def test_ringmodulator(self):
        print("test_ringmodulator")

        class RingModulator(SubCircuitFactory):
            NAME = 'RingModulator'
            NODES = ('input_plus', 'input_minus',
                     'carrier_plus', 'carrier_minus',
                     'output_plus', 'output_minus')

            def __init__(self,
                         outer_inductance,
                         inner_inductance,
                         coupling,
                         diode_model,
                         ):
                super().__init__()

                input_inductor = self.L('input', 'input_plus', 'input_minus', outer_inductance)
                top_inductor = self.L('input_top', 'input_top', 'carrier_plus', inner_inductance)
                bottom_inductor = self.L('input_bottom', 'carrier_plus', 'input_bottom', inner_inductance)
                self.CoupledInductor('input_top', input_inductor.name, top_inductor.name, coupling)
                self.CoupledInductor('input_bottom', input_inductor.name, bottom_inductor.name, coupling)

                self.X('D1', diode_model, 'input_top', 'output_top')
                self.X('D2', diode_model, 'output_top', 'input_bottom')
                self.X('D3', diode_model, 'input_bottom', 'output_bottom')
                self.X('D4', diode_model, 'output_bottom', 'input_top')

                top_inductor = self.L('output_top', 'output_top', 'carrier_minus', inner_inductance)
                bottom_inductor = self.L('output_bottom', 'carrier_minus', 'output_bottom', inner_inductance)
                output_inductor = self.L('output', 'output_plus', 'output_minus', outer_inductance)
                self.CoupledInductor('output_top', output_inductor.name, top_inductor.name, coupling)
                self.CoupledInductor('output_bottom', output_inductor.name, bottom_inductor.name, coupling)

    def test_diode_characteristic_curve(self):
        print("test_diode_characteristic_curve")

        logger = Logging.setup_logging()

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        circuit = Circuit('Diode Characteristic Curve')

        circuit.include(spice_library['1N4148'])

        circuit.V('input', 'in', circuit.gnd, 10)
        circuit.R(1, 'in', 'out', 1)  # not required for simulation
        circuit.X('D1', '1N4148', 'out', circuit.gnd)

        # Fixme: Xyce ???
        # degres celsius
        temperatures = [0, 25, 100]
        analyses = {}
        for temperature in temperatures:
            simulator = circuit.simulator(temperature=temperature,
                                          nominal_temperature=temperature)
            analysis = simulator.dc(Vinput=slice(-2, 5, .01))
            analyses[float(temperature)] = analysis

        silicon_forward_voltage_threshold = .7

        shockley_diode = ShockleyDiode(Is=4e-9, degree=25)

        def two_scales_tick_formatter(value, position):
            if value >= 0:
                return '{} mA'.format(value)
            else:
                return '{} nA'.format(value / 100)

        formatter = ticker.FuncFormatter(two_scales_tick_formatter)

        figure, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

        ax1.set_title('1N4148 Characteristic Curve ')
        ax1.set_xlabel('Voltage [V]')
        ax1.set_ylabel('Current')
        ax1.grid()
        ax1.set_xlim(-2, 2)
        ax1.axvspan(-2, 0, facecolor='green', alpha=.2)
        ax1.axvspan(0, silicon_forward_voltage_threshold, facecolor='blue', alpha=.1)
        ax1.axvspan(silicon_forward_voltage_threshold, 2, facecolor='blue', alpha=.2)
        ax1.set_ylim(-500, 750)  # Fixme: round
        ax1.yaxis.set_major_formatter(formatter)
        Vd = analyses[25].out
        # compute scale for reverse and forward region
        forward_region = Vd >= 0
        reverse_region = np.invert(forward_region)
        scale = reverse_region * 1e11 + forward_region * 1e3
        for temperature in temperatures:
            analysis = analyses[float(temperature)]
            ax1.plot(Vd, - analysis.Vinput * scale)
        ax1.plot(Vd, shockley_diode.I(Vd) * scale, 'black')
        ax1.legend(['@ {} °C'.format(temperature)
                    for temperature in temperatures] + ['Shockley Diode Model Is = 4 nA'],
                   loc=(.02, .8))
        ax1.axvline(x=0, color='black')
        ax1.axhline(y=0, color='black')
        ax1.axvline(x=silicon_forward_voltage_threshold, color='red')
        ax1.text(-1, -100, 'Reverse Biased Region', ha='center', va='center')
        ax1.text(1, -100, 'Forward Biased Region', ha='center', va='center')

        ax2.set_title('Resistance @ 25 °C')
        ax2.grid()
        ax2.set_xlim(-2, 3)
        ax2.axvspan(-2, 0, facecolor='green', alpha=.2)
        ax2.axvspan(0, silicon_forward_voltage_threshold, facecolor='blue', alpha=.1)
        ax2.axvspan(silicon_forward_voltage_threshold, 3, facecolor='blue', alpha=.2)
        analysis = analyses[25]
        static_resistance = -analysis.out / analysis.Vinput
        dynamic_resistance = np.diff(-analysis.out) / np.diff(analysis.Vinput)
        ax2.semilogy(analysis.out, static_resistance, basey=10)
        ax2.semilogy(analysis.out[10:-1], dynamic_resistance[10:], basey=10)
        ax2.axvline(x=0, color='black')
        ax2.axvline(x=silicon_forward_voltage_threshold, color='red')
        ax2.axhline(y=1, color='red')
        ax2.text(-1.5, 1.1, 'R limitation = 1 Ω', color='red')
        ax2.legend(['{} Resistance'.format(x) for x in ('Static', 'Dynamic')], loc=(.05, .2))
        ax2.set_xlabel('Voltage [V]')
        ax2.set_ylabel('Resistance [Ω]')

        plt.tight_layout()
        plt.show()

    def test_diode_recovery_time(self):
        print("test_diode_recovery_time")

        logger = Logging.setup_logging()

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        dc_offset = 1
        ac_amplitude = 100

        circuit = Circuit('Diode')
        circuit.include(spice_library['BAV21'])
        # Fixme: Xyce: Device model BAV21: Illegal parameter(s) given for level 1 diode: IKF
        source = circuit.V('input', 'in', circuit.gnd, dc_offset)
        circuit.R(1, 'in', 'out', 1)
        circuit.D('1', 'out', circuit.gnd, model='BAV21')

        quiescent_points = []

        for voltage in (dc_offset - ac_amplitude, dc_offset, dc_offset + ac_amplitude):
            source.dc_value = voltage
            simulator = circuit.simulator(temperature=25, nominal_temperature=25)
            analysis = simulator.operating_point()
            # Fixme: handle unit
            quiescent_voltage = float(analysis.out)
            quiescent_current = - float(analysis.Vinput)
            quiescent_points.append(dict(voltage=voltage,
                                         quiescent_voltage=quiescent_voltage,
                                         quiescent_current=quiescent_current))
            print("Quiescent Point {:.1f} mV {:.1f} mA".format(quiescent_voltage * 1e3, quiescent_current * 1e3))

        dynamic_resistance = ((quiescent_points[0]['quiescent_voltage'] -
                               quiescent_points[-1]['quiescent_voltage'])
                              /
                              (quiescent_points[0]['quiescent_current'] -
                               quiescent_points[-1]['quiescent_current']))

        circuit = Circuit('Diode')
        circuit.include(spice_library['BAV21'])
        circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd,
                                        dc_offset=dc_offset, offset=dc_offset,
                                        amplitude=ac_amplitude)
        R = circuit.R(1, 'in', 'out', 1)
        circuit.D('1', 'out', circuit.gnd, model='BAV21')

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.ac(start_frequency=10, stop_frequency=1, number_of_points=10,
                                variation='dec')

        figure, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 10))

        # Fixme: handle unit in plot (scale and legend)
        ax1.semilogx(analysis.frequency, np.absolute(analysis.out) * 1e3)
        ax1.grid(True)
        ax1.grid(True, which='minor')
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Vd [mV]")

        current = (analysis['in'] - analysis.out) / float(R.resistance)
        ax2.semilogx(analysis.frequency, np.absolute(analysis.out / current))
        ax2.grid(True)
        ax2.grid(True, which='minor')
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel('Rd [Ω]')

        frequency = 1

        circuit = Circuit('Diode')
        circuit.include(spice_library['BAV21'])
        # source = circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd,
        #                             dc_offset=dc_offset, offset=dc_offset,
        #                             amplitude=ac_amplitude,
        #                             frequency=frequency)
        source = circuit.PulseVoltageSource('input', 'in', circuit.gnd,
                                            initial_value=dc_offset - ac_amplitude,
                                            pulsed_value=dc_offset + ac_amplitude,
                                            pulse_width=frequency.period / 2, period=frequency.period)
        circuit.R(1, 'in', 'out', 1)
        circuit.D('1', 'out', circuit.gnd, model='BAV21')

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=source.period / 1e3, end_time=source.period * 4)

        # Fixme: axis, x scale
        # plot(analysis['in'] - dc_offset + quiescent_points[0]['quiescent_voltage'])
        # plot(analysis.out)
        ax3.plot(analysis.out.abscissa * 1e6, analysis.out)
        ax3.legend(('Vin [V]', 'Vout [V]'), loc=(.8, .8))
        ax3.grid()
        ax3.set_xlabel('t [μs]')
        ax3.set_ylabel('[V]')
        # ax3.set_ylim(.5, 1 + ac_amplitude + .1)

        plt.tight_layout()
        plt.show()

    def test_rectification(self):
        print("test_rectification")

        logger = Logging.setup_logging()

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        figure1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))

        circuit = Circuit('half-wave rectification')
        circuit.include(spice_library['1N4148'])
        source = circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd, amplitude=10, frequency=50)
        circuit.X('D1', '1N4148', 'in', 'output')
        circuit.R('load', 'output', circuit.gnd, 100)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=source.period / 200, end_time=source.period * 2)

        ax1.set_title('Half-Wave Rectification')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Voltage [V]')
        ax1.grid()
        ax1.plot(analysis['in'])
        ax1.plot(analysis.output)
        ax1.legend(('input', 'output'), loc=(.05, .1))
        ax1.set_ylim(float(-source.amplitude * 1.1), float(source.amplitude * 1.1))

        circuit.C('1', 'output', circuit.gnd, 1)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=source.period / 200, end_time=source.period * 2)

        ax2.set_title('Half-Wave Rectification with filtering')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Voltage [V]')
        ax2.grid()
        ax2.plot(analysis['in'])
        ax2.plot(analysis.output)
        ax2.legend(('input', 'output'), loc=(.05, .1))
        ax2.set_ylim(float(-source.amplitude * 1.1), float(source.amplitude * 1.1))

        circuit = Circuit('half-wave rectification')
        circuit.include(spice_library['1N4148'])
        source = circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd, amplitude=10, frequency=50)
        circuit.X('D1', '1N4148', 'in', 'output_plus')
        circuit.R('load', 'output_plus', 'output_minus', 100)
        circuit.X('D2', '1N4148', 'output_minus', circuit.gnd)
        circuit.X('D3', '1N4148', circuit.gnd, 'output_plus')
        circuit.X('D4', '1N4148', 'output_minus', 'in')

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=source.period / 200, end_time=source.period * 2)

        ax3.set_title('Full-Wave Rectification')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Voltage [V]')
        ax3.grid()
        ax3.plot(analysis['in'])
        ax3.plot(analysis.output_plus - analysis.output_minus)
        ax3.legend(('input', 'output'), loc=(.05, .1))
        ax3.set_ylim(float(-source.amplitude * 1.1), float(source.amplitude * 1.1))

        circuit.C('1', 'output_plus', 'output_minus', 1)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=source.period / 200, end_time=source.period * 2)

        ax4.set_title('Full-Wave Rectification with filtering')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Voltage [V]')
        ax4.grid()
        ax4.plot(analysis['in'])
        ax4.plot(analysis.output_plus - analysis.output_minus)
        ax4.legend(('input', 'output'), loc=(.05, .1))
        ax4.set_ylim(float(-source.amplitude * 1.1), float(source.amplitude * 1.1))

        plt.tight_layout()

        circuit = Circuit('115/230V Rectifier')
        circuit.include(spice_library['1N4148'])
        on_115 = True  # switch to select 115 or 230V
        if on_115:
            node_230 = circuit.gnd
            node_115 = 'node_115'
            amplitude = 115
        else:
            node_230 = 'node_230'
            node_115 = circuit.gnd
            amplitude = 230

        source = circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd, amplitude=amplitude,
                                                 frequency=50)  # Fixme: rms
        circuit.X('D1', '1N4148', 'in', 'output_plus')
        circuit.X('D3', '1N4148', node_230, 'output_plus')
        circuit.X('D2', '1N4148', 'output_minus', node_230)
        circuit.X('D4', '1N4148', 'output_minus', 'in')
        circuit.C('1', 'output_plus', node_115, 1)
        circuit.C('2', node_115, 'output_minus', 1)
        circuit.R('load', 'output_plus', 'output_minus', 10)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)

        if on_115:
            simulator.initial_condition(node_115=0)
        analysis = simulator.transient(step_time=source.period / 200, end_time=source.period * 2)

        figure2, ax = plt.subplots(figsize=(20, 10))
        ax.set_title('115/230V Rectifier')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [V]')
        ax.grid()
        ax.plot(analysis['in'])
        ax.plot(analysis.output_plus - analysis.output_minus)
        ax.legend(('input', 'output'), loc=(.05, .1))
        # ax.set_ylim(float(-source.amplitude*1.1), float(source.amplitude*1.1))

        plt.tight_layout()

        plt.show()

    def test_ring_modulator(self):
        print("test_ring_modulator")

        logger = Logging.setup_logging()

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        ####################################################################################################

        class RingModulator(SubCircuitFactory):
            NAME = 'RingModulator'
            NODES = ('input_plus', 'input_minus',
                     'carrier_plus', 'carrier_minus',
                     'output_plus', 'output_minus')

            ##############################################

            def __init__(self,
                         outer_inductance,
                         inner_inductance,
                         coupling,
                         diode_model,
                         ):
                super().__init__()

                input_inductor = self.L('input', 'input_plus', 'input_minus', outer_inductance)
                top_inductor = self.L('input_top', 'input_top', 'carrier_plus', inner_inductance)
                bottom_inductor = self.L('input_bottom', 'carrier_plus', 'input_bottom', inner_inductance)
                self.CoupledInductor('input_top', input_inductor.name, top_inductor.name, coupling)
                self.CoupledInductor('input_bottom', input_inductor.name, bottom_inductor.name, coupling)

                self.X('D1', diode_model, 'input_top', 'output_top')
                self.X('D2', diode_model, 'output_top', 'input_bottom')
                self.X('D3', diode_model, 'input_bottom', 'output_bottom')
                self.X('D4', diode_model, 'output_bottom', 'input_top')

                top_inductor = self.L('output_top', 'output_top', 'carrier_minus', inner_inductance)
                bottom_inductor = self.L('output_bottom', 'carrier_minus', 'output_bottom', inner_inductance)
                output_inductor = self.L('output', 'output_plus', 'output_minus', outer_inductance)
                self.CoupledInductor('output_top', output_inductor.name, top_inductor.name, coupling)
                self.CoupledInductor('output_bottom', output_inductor.name, bottom_inductor.name, coupling)

        circuit = Circuit('Ring Modulator')

        modulator = circuit.SinusoidalVoltageSource('modulator', 'in', circuit.gnd, amplitude=1,
                                                    frequency=1)
        carrier = circuit.SinusoidalVoltageSource('carrier', 'carrier', circuit.gnd, amplitude=10,
                                                  frequency=100)
        circuit.R('in', 'in', 1, 50)
        circuit.R('carrier', 'carrier', 2, 50)

        circuit.include(spice_library['1N4148'])
        circuit.subcircuit(RingModulator(outer_inductance=1,
                                         inner_inductance=1,
                                         coupling=.99,
                                         diode_model='1N4148',
                                         ))
        circuit.X('ring_modulator', 'RingModulator',
                  1, circuit.gnd,
                  2, circuit.gnd,
                  'output', circuit.gnd,
                  )

        # outer_inductance = .01
        # inner_inductance = .0025
        # coupling = .9
        # diode_model = '1N4148'
        # input_inductor = circuit.L('input', 1, circuit.gnd, outer_inductance)
        # top_inductor = circuit.L('input_top', 'input_top', 'carrier', inner_inductance)
        # bottom_inductor = circuit.L('input_bottom', 'input_bottom', 'carrier', inner_inductance)
        # circuit.CoupledInductor('input_top', input_inductor.name, top_inductor.name, coupling)
        # circuit.CoupledInductor('input_bottom', input_inductor.name, bottom_inductor.name, coupling)
        # circuit.X('D1', diode_model, 'input_top', 'output_top')
        # circuit.X('D2', diode_model, 'output_top', 'input_bottom')
        # circuit.X('D3', diode_model, 'input_bottom', 'output_bottom')
        # circuit.X('D4', diode_model, 'output_bottom', 'input_top')
        # top_inductor = circuit.L('output_top', 'output_top', circuit.gnd, inner_inductance)
        # bottom_inductor = circuit.L('output_bottom', 'output_bottom', circuit.gnd, inner_inductance)
        # output_inductor = circuit.L('output', 'output', circuit.gnd, outer_inductance)
        # circuit.CoupledInductor('output_top', output_inductor.name, top_inductor.name, coupling)
        # circuit.CoupledInductor('output_bottom', output_inductor.name, bottom_inductor.name, coupling)

        circuit.R('load', 'output', circuit.gnd, 1)

        ### simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        ### # simulator.initial_condition(input_top=0, input_bottom=0, output_top=0, output_bottom=0)
        ### analysis = simulator.transient(step_time=modulator.period/1000, end_time=modulator.period)
        ###
        ### figure = plt.figure(1, (20, 10))
        ### plt.title('Ring Modulator')
        ### plt.xlabel('Time [s]')
        ### plt.ylabel('Voltage [V]')
        ### plt.grid()
        ### plot(analysis['Vmodulator'])
        ### plot(analysis['Vcarrier'])
        ### # plot(analysis['output'])
        ### plt.legend(('modulator', 'carrier', 'output'), loc=(.05,.1))

        plt.show()

    def test_voltage_multiplier(self):
        print("test_voltage_multiplier")

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        circuit = Circuit('Voltage Multiplier')
        circuit.include(spice_library[''])
        source = circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd, amplitude=10, frequency=50)

        multiplier = 5

        for i in range(multiplier):
            if i:
                top_node = i - 1
            else:
                top_node = 'in'
            midlle_node, bottom_node = i + 1, i
            circuit.C(i, top_node, midlle_node, 1)
            circuit.X(i, '1N4148', midlle_node, bottom_node)

        circuit.R(1, multiplier, multiplier + 1, 1)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=source.period / 200, end_time=source.period * 20)

        figure, ax = plt.subplots(figsize=(20, 10))

        ax.set_title('Voltage Multiplier')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [V]')
        ax.grid()

        # Fixme: axis vs axe ...
        ax.plot(analysis['in'])
        for i in range(1, multiplier + 1):
            y = analysis[str(i)]
            if i & 1:  # for odd multiplier the ground is permuted
                y -= analysis['in']
            ax.plot(y)

        # ax.axhline(-multiplier*source.amplitude)
        ax.set_ylim(float(-multiplier * 1.1 * source.amplitude), float(1.1 * source.amplitude))
        ax.legend(['input'] + ['*' + str(i) for i in range(1, multiplier + 1)],
                  loc=(.2, .8))

        plt.tight_layout()
        plt.show()

    def test_zener_characteristic_curve(self):
        print("test_zener_characteristic_curve")

        # r# This example shows how to simulate and plot the characteristic curve of a Zener diode.

        logger = Logging.setup_logging()

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        # f# circuit_macros('zener-diode-characteristic-curve-circuit.m4')

        circuit = Circuit('Diode DC Curve')

        circuit.include(spice_library['1N4148'])
        # 1N5919B: 5.6 V, 3.0 W Zener Diode Voltage Regulator
        circuit.include(spice_library['d1n5919brl'])

        circuit.V('input', 'in', circuit.gnd, 10)
        circuit.R(1, 'in', 'out', 1)  # not required for simulation
        # circuit.X('D1', '1N4148', 'out', circuit.gnd)
        circuit.X('DZ1', 'd1n5919brl', 'out', circuit.gnd)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.dc(Vinput=slice(-10, 2, .05))  # 10mV

        figure, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

        zener_part = analysis.out <= -5.4
        # compute derivate
        # fit linear part

        ax1.grid()
        # Fixme: scale
        ax1.plot(analysis.out, -analysis.Vinput * 1000)
        ax1.axvline(x=0, color='black')
        ax1.axvline(x=-5.6, color='red')
        ax1.axvline(x=1, color='red')
        ax1.legend(('Diode curve',), loc=(.1, .8))
        ax1.set_xlabel('Voltage [V]')
        ax1.set_ylabel('Current [mA]')

        ax2.grid()
        # Fixme:
        # U = RI   R = U/I
        dynamic_resistance = np.diff(-analysis.out) / np.diff(analysis.Vinput)
        # ax2.plot(analysis.out[:-1], dynamic_resistance/1000)
        ax2.semilogy(analysis.out[10:-1], dynamic_resistance[10:], basey=10)
        ax2.axvline(x=0, color='black')
        ax2.axvline(x=-5.6, color='red')
        ax2.legend(('Dynamic Resistance',), loc=(.1, .8))
        ax2.set_xlabel('Voltage [V]')
        ax2.set_ylabel('Dynamic Resistance [Ohm]')

        # coefficients = np.polyfit(analysis.out[zener_part], dynamic_resistance[zener_part], deg=1)
        # x = np.array((min(analysis.out[zener_part]), max(analysis.out[zener_part])))
        # y = coefficients[0]*x + coefficients[1]
        # axe.semilogy(x, y, 'red')

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'zener-characteristic-curve.png')

    def test_three_phased_current_y_and_delta_configurations(self):
        print("test_three_phased_current_y_and_delta_configurations")

        ####################################################################################################

        # r#
        # r# =================================================
        # r#  Three-phased Current: Y and Delta configurations
        # r# =================================================
        # r#
        # r# This examples shows the computation of the voltage for the Y and Delta configurations.
        # r#

        ####################################################################################################

        import math

        import numpy as np
        import matplotlib.pyplot as plt

        # r# Let use an European 230 V / 50 Hz electric network.

        frequency = 50
        w = frequency.pulsation
        period = frequency.period

        rms_mono = 230
        amplitude_mono = rms_mono * math.sqrt(2)

        # r# The phase voltages in Y configuration are dephased of :math:`\frac{2\pi}{3}`:
        # r#
        # r# .. math::
        # r#  V_{L1 - N} = V_{pp} \cos \left( \omega t \right) \\
        # r#  V_{L2 - N} = V_{pp} \cos \left( \omega t - \frac{2\pi}{3} \right) \\
        # r#  V_{L3 - N} = V_{pp} \cos \left( \omega t - \frac{4\pi}{3} \right)
        # r#
        # r# We rewrite them in complex notation:
        # r#
        # r# .. math::
        # r#  V_{L1 - N} = V_{pp} e^{j\omega t} \\
        # r#  V_{L2 - N} = V_{pp} e^{j \left(\omega t - \frac{2\pi}{3} \right) } \\
        # r#  V_{L3 - N} = V_{pp} e^{j \left(\omega t - \frac{4\pi}{3} \right) }

        t = np.linspace(0, 3 * float(period), 1000)
        L1 = amplitude_mono * np.cos(t * w)
        L2 = amplitude_mono * np.cos(t * w - 2 * math.pi / 3)
        L3 = amplitude_mono * np.cos(t * w - 4 * math.pi / 3)

        # r# From these expressions, we compute the voltage in delta configuration using trigonometric identities :
        # r#
        # r# .. math::
        # r#   V_{L1 - L2} = V_{L1} \sqrt{3} e^{j \frac{\pi}{6} } \\
        # r#   V_{L2 - L3} = V_{L2} \sqrt{3} e^{j \frac{\pi}{6} } \\
        # r#   V_{L3 - L1} = V_{L3} \sqrt{3} e^{j \frac{\pi}{6} }
        # r#
        # r# In comparison to the Y configuration, the voltages in delta configuration are magnified by
        # r# a factor :math:`\sqrt{3}` and dephased of :math:`\frac{\pi}{6}`.
        # r#
        # r# Finally we rewrite them in temporal notation:
        # r#
        # r# .. math::
        # r#  V_{L1 - L2} = V_{pp} \sqrt{3} \cos \left( \omega t + \frac{\pi}{6} \right) \\
        # r#  V_{L2 - L3} = V_{pp} \sqrt{3} \cos \left( \omega t - \frac{\pi}{2} \right) \\
        # r#  V_{L3 - L1} = V_{pp} \sqrt{3} \cos \left( \omega t - \frac{7\pi}{6} \right)

        rms_tri = math.sqrt(3) * rms_mono
        amplitude_tri = rms_tri * math.sqrt(2)

        L12 = amplitude_tri * np.cos(t * w + math.pi / 6)
        L23 = amplitude_tri * np.cos(t * w - math.pi / 2)
        L31 = amplitude_tri * np.cos(t * w - 7 * math.pi / 6)

        # r# Now we plot the waveforms:
        figure, ax = plt.subplots(figsize=(20, 10))
        ax.plot(
            t, L1, t, L2, t, L3,
            t, L12, t, L23, t, L31,
            # t, L1-L2, t, L2-L3, t, L3-L1,
        )
        ax.grid()
        ax.set_title('Three-phase electric power: Y and Delta configurations (230V Mono/400V Tri 50Hz Europe)')
        ax.legend(
            (
                'L1-N',
                'L2-N',
                'L3-N',
                'L1-L2',
                'L2-L3',
                'L3-L1'
            ),
            loc=(.7, .5),
        )
        ax.set_xlabel('t [s]')
        ax.set_ylabel('[V]')
        ax.axhline(y=rms_mono, color='blue')
        ax.axhline(y=-rms_mono, color='blue')
        ax.axhline(y=rms_tri, color='blue')
        ax.axhline(y=-rms_tri, color='blue')

        plt.show()

        # f# save_figure('figure', 'three-phase.png')

    def test_low_pass_rc_filter(self):
        print("test_low_pass_rc_filter")

        # r# This example shows a low-pass RC Filter.

        ####################################################################################################

        import math
        import numpy as np
        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Plot.BodeDiagram import bode_diagram
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        # f# circuit_macros('low-pass-rc-filter.m4')

        circuit = Circuit('Low-Pass RC Filter')

        circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd, amplitude=1)
        R1 = circuit.R(1, 'in', 'out', 1)
        C1 = circuit.C(1, 'out', circuit.gnd, 1)

        # r# The break frequency is given by :math:`f_c = \frac{1}{2 \pi R C}`

        break_frequency = 1 / (2 * math.pi * float(R1.resistance * C1.capacitance))
        print("Break frequency = {:.1f} Hz".format(break_frequency))
        # o#

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.ac(start_frequency=1, stop_frequency=1, number_of_points=10,
                                variation='dec')
        # print(analysis.out)

        # r# We plot the Bode diagram.

        figure, axes = plt.subplots(2, figsize=(20, 10))
        plt.title("Bode Diagram of a Low-Pass RC Filter")
        bode_diagram(axes=axes,
                     frequency=analysis.frequency,
                     gain=20 * np.log10(np.absolute(analysis.out)),
                     phase=np.angle(analysis.out, deg=False),
                     marker='.',
                     color='blue',
                     linestyle='-',
                     )

        for ax in axes:
            ax.axvline(x=break_frequency, color='red')

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'low-pass-rc-filter-bode-diagram.png')

    def test_rlc_filter(self):
        print("test_rlc_filter")

        # r# ============
        # r#  RLC Filter
        # r# ============

        # r# This example illustrates RLC Filters.

        ####################################################################################################

        import math

        import numpy as np
        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Plot.BodeDiagram import bode_diagram
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        # r# We define four low-pass RLC filters with the following factor of quality: .5, 1, 2 and 4.

        # f# circuit_macros('low-pass-rlc-filter.m4')

        circuit1 = Circuit('Four double-pole Low-Pass RLC Filter')

        inductance = 10
        capacitance = 1

        circuit1.SinusoidalVoltageSource('input', 'in', circuit1.gnd, amplitude=1)
        # ?# pulse 0 5 10 ms
        # Q = .5
        circuit1.R(1, 'in', 1, 200)
        circuit1.L(1, 1, 'out5', inductance)
        circuit1.C(1, 'out5', circuit1.gnd, capacitance)
        # Q = 1
        circuit1.R(2, 'in', 2, 100)
        circuit1.L(2, 2, 'out1', inductance)
        circuit1.C(2, 'out1', circuit1.gnd, capacitance)
        # Q = 2
        circuit1.R(3, 'in', 3, 50)
        circuit1.L(3, 3, 'out2', inductance)
        circuit1.C(3, 'out2', circuit1.gnd, capacitance)
        # Q = 4
        R4 = circuit1.R(4, 'in', 4, 25)
        circuit1.L(4, 4, 'out4', inductance)
        circuit1.C(4, 'out4', circuit1.gnd, capacitance)

        # r# We perform an AC analysis.

        simulator1 = circuit1.simulator(temperature=25,
                                        nominal_temperature=25)

        analysis1 = simulator1.ac(start_frequency=100,
                                  stop_frequency=10,
                                  number_of_points=100,
                                  variation='dec')

        # r# The resonant frequency is given by
        # r#
        # r# .. math::
        # r#
        # r#     f_0 = 2 \pi \omega_0 = \frac{1}{2 \pi \sqrt{L C}}
        # r#
        # r# and the factor of quality by
        # r#
        # r# .. math::
        # r#
        # r#     Q = \frac{1}{R} \sqrt{\frac{L}{C}} = \frac{1}{RC \omega_0}
        # r#

        resonant_frequency = 1 / (2 * math.pi * math.sqrt(inductance * capacitance))
        quality_factor = 1 / R4.resistance * math.sqrt(inductance / capacitance)
        print("Resonant frequency = {:.1f} Hz".format(resonant_frequency))
        print("Factor of quality = {:.1f}".format(quality_factor))
        # o#

        # r# We plot the Bode diagram of the four filters.

        figure, axes = plt.subplots(2, figsize=(20, 10))

        plt.title("Bode Diagrams of RLC Filters")

        for out in ('out5', 'out1', 'out2', 'out4'):
            bode_diagram(axes=axes,
                         frequency=analysis1.frequency,
                         gain=20 * np.log10(np.absolute(analysis1[out])),
                         phase=np.angle(analysis1[out], deg=False),
                         marker='.',
                         color='blue',
                         linestyle='-',
                         )

        for axe in axes:
            axe.axvline(x=resonant_frequency, color='red')

        ####################################################################################################

        # r# We define a pass-band RLC filter with a quality's factor of 4.

        # f# circuit_macros('pass-band-rlc-filter.m4')

        circuit2 = Circuit('Pass-Band RLC Filter')

        circuit2.SinusoidalVoltageSource('input', 'in', circuit2.gnd, amplitude=1)
        circuit2.L(1, 'in', 2, inductance)
        circuit2.C(1, 2, 'out', capacitance)
        circuit2.R(1, 'out', circuit2.gnd, 25)

        simulator2 = circuit2.simulator(temperature=25, nominal_temperature=25)

        analysis2 = simulator2.ac(start_frequency=100, stop_frequency=10, number_of_points=100,
                                  variation='dec')

        bode_diagram(axes=axes,
                     frequency=analysis2.frequency,
                     gain=20 * np.log10(np.absolute(analysis2.out)),
                     phase=np.angle(analysis2.out, deg=False),
                     marker='.',
                     color='magenta',
                     linestyle='-',
                     )

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'rlc-filter-bode-diagram.png')

    def test_millman_theorem(self):
        print("test_millman_theorem")

        ####################################################################################################

        # r# ===================
        # r#  Millman's theorem
        # r# ===================
        # r#
        # r# Millman's theorem is a method to compute the voltage of a node in such circuits:

        # f# circuit_macros('millman-theorem.m4')

        # r# The voltage at node A is:
        # r#
        # r# .. math::
        # r#
        # r#     V_A = \frac{\sum \frac{V_i}{R_i}}{\sum \frac{1}{R_i}}
        # r#
        # r# We can extend this theorem to branches with current sources:
        # r#
        # r# .. math::
        # r#
        # r#     V_A = \frac{\sum \frac{V_i}{R_i} + \sum I_k}{\sum \frac{1}{R_i}}
        # r#
        # r# Note voltage sources can be null and resistances in current's branches don't change the denominator.

        # f# circuit_macros('millman-theorem-with-current-source.m4')

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        import numpy as np

        ####################################################################################################

        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        circuit = Circuit("Millman's theorem")

        number_of_branches = 3
        for i in range(1, number_of_branches + 1):
            circuit.V('input%u' % i, i, circuit.gnd, i)
            circuit.R(i, i, 'A', i)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()

        node_A = analysis.A
        print('Node {}: {:5.2f} V'.format(str(node_A), float(node_A)))
        # o#

        branch_voltages = np.arange(1, number_of_branches + 1)
        branch_resistances = branch_voltages * float(10^(-3))
        conductances = 1 / branch_resistances
        voltage_A = np.sum(branch_voltages * conductances) / np.sum(conductances)
        print('V(A) = {:5.2f} V'.format(voltage_A))
        # o#

        # with current sources
        for i in range(1, number_of_branches + 1):
            ii = number_of_branches + i
            circuit.I('input%u' % i, circuit.gnd, ii, 100 * i)
            circuit.R(ii, ii, 'A', i)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()

        node_A = analysis.A
        print('Node {}: {:5.2f} V'.format(str(node_A), float(node_A)))
        # o#

        branch_currents = np.arange(1, number_of_branches + 1) * float(100*10^(-6))
        voltage_A += np.sum(branch_currents) / np.sum(conductances)
        print('V(A) = {:5.2f} V'.format(voltage_A))
        # o#

    def test_thevenin_and_norton_theorem(self):
        print("test_thevenin_and_norton_theorem")

        ####################################################################################################

        # r# ============================
        # r#  Thévenin and Norton Theorem
        # r# ============================

        # r# The Thévenin's theorem holds that:
        # r#
        # r#  * Any linear electrical network with voltage and current sources and only resistances can be
        # r#    replaced at terminals A-B by an equivalent voltage source Vth in series connection with an
        # r#    equivalent resistance Rth.
        # r#
        # r#  * This equivalent voltage Vth is the voltage obtained at terminals A-B of the network with
        # r#    terminals A-B open circuited.
        # r#
        # r#  * This equivalent resistance Rth is the resistance obtained at terminals A-B of the network
        # r#    with all its independent current sources open circuited and all its independent voltage
        # r#    sources short circuited.
        # r#
        # r# The Norton's theorem holds that:
        # r#
        # r#  * Any linear electrical network with voltage and current sources and only resistances can be
        # r#    replaced at terminals A-B by an equivalent current source INO in parallel connection with an
        # r#    equivalent resistance Rno.
        # r#
        # r#  * This equivalent current Ino is the current obtained at terminals A-B of the network with
        # r#    terminals A-B short circuited.
        # r#
        # r#  * This equivalent resistance Rno is the resistance obtained at terminals A-B of the network
        # r#    with all its voltage sources short circuited and all its current sources open circuited.
        # r#
        # r# The Norton's theorem is the dual of the Thévenin's therorem and both are related by
        # r# these equations:
        # r#
        # r#  .. math::
        # r#
        # r#       \begin{align}
        # r#         R_{no} & = R_{th} \\
        # r#         I_{no} & = V_{th} / R_{th} \\
        # r#         V_{th} & = I_{No} R_{no}
        # r#       \end{align}

        # f# circuit_macros('thevenin-norton-theorem.m4')

        # r# In circuit theory terms, these theorems allows any one-port network to be reduced to a single
        # r# voltage or current source and a single impedance.
        # r#
        # r# For AC circuits these theorems can be applied to reactive impedances as well as resistances.

        # ?# These theorems also applies to frequency domain AC circuits consisting of reactive and resistive
        # ?# impedances.

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        thevenin_circuit = Circuit('Thévenin Representation')

        thevenin_circuit.V('input', 1, thevenin_circuit.gnd, 10)
        thevenin_circuit.R('generator', 1, 'load', 10)
        thevenin_circuit.R('load', 'load', thevenin_circuit.gnd, 1)

        simulator = thevenin_circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()

        load_node = analysis.load
        print('Node {}: {:5.2f} V'.format(str(load_node), float(load_node)))
        # o#

        norton_circuit = Circuit('Norton Representation')

        norton_circuit.I('input', norton_circuit.gnd, 'load',
                         thevenin_circuit.Vinput.dc_value / thevenin_circuit.Rgenerator.resistance)
        norton_circuit.R('generator', 'load', norton_circuit.gnd, thevenin_circuit.Rgenerator.resistance)
        norton_circuit.R('load', 'load', norton_circuit.gnd, thevenin_circuit.Rload.resistance)

        simulator = norton_circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()

        load_node = analysis.load
        print('Node {}: {:5.2f} V'.format(str(load_node), float(load_node)))
        # o#

    def test_voltage_and_current_divider(self):
        print("test_voltage_and_current_divider")

        ####################################################################################################

        # r# =============================
        # r#  Voltage and Current Divider
        # r# =============================

        # r# This circuit is a fundamental block in electronic that permits to scale a voltage by an
        # r# impedance ratio:

        # f# circuit_macros('voltage-divider.m4')

        # r# The relation between the input and ouput voltage is:
        # r#
        # r# .. math::
        # r#
        # r#     \frac{V_{out}}{V_{in}} = \frac{R_2}{R_1 + R_2}
        # r#
        # r# This equation holds for any impedances like resistance, capacitance, inductance, etc.

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        circuit = Circuit('Voltage Divider')

        circuit.V('input', 1, circuit.gnd, 10)
        circuit.R(1, 1, 2, 2)
        circuit.R(2, 2, circuit.gnd, 1)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()

        for node in analysis.nodes.values():
            print('Node {}: {:5.2f} V'.format(str(node), float(node)))  # Fixme: format value + unit
        # o#

        ####################################################################################################

        # r# Similarly we can build a circuit that scale a current by an impedance ratio:

        # f# circuit_macros('current-divider.m4')

        # r# The relation between the input and ouput current is:
        # r#
        # r# .. math::
        # r#
        # r#     \frac{I_{out}}{I_{in}} = \frac{R_1}{R_1 + R_2}
        # r#
        # r# Note the role of R1 and R2 is exchanged.
        # r#
        # r# This equation holds for any impedances like resistance, capacitance, inductance, etc.

        ####################################################################################################

        circuit = Circuit('Current Divider')

        circuit.I('input', 1, circuit.gnd, 1)  # Fixme: current value
        circuit.R(1, 1, circuit.gnd, 2)
        circuit.R(2, 1, circuit.gnd, 1)

        for resistance in (circuit.R1, circuit.R2):
            resistance.minus.add_current_probe(circuit)  # to get positive value

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()

        # Fixme: current over resistor
        for node in analysis.branches.values():
            print('Node {}: {:5.2f} A'.format(str(node), float(node)))  # Fixme: format value + unit
        # o#

    def test_simulation_using_external_sources(self):
        print("test_simulation_using_external_sources")

        ####################################################################################################

        # r#
        # r# ===================================
        # r#  Simulation using External Sources
        # r# ===================================
        # r#
        # r# This example explains how to plug a voltage source from Python to NgSpice.
        # r#

        ####################################################################################################

        # Fixme: Travis CI macOS
        #
        # Error on line 2 :
        #   vinput input 0 dc 0 external
        #   parameter value out of range or the wrong type
        #
        # Traceback (most recent call last):
        #     analysis = simulator.transient(step_time=period/200, end_time=period*2)
        #   File "/usr/local/lib/python3.7/site-packages/PySpice/Spice/NgSpice/Shared.py", line 1145, in load_circuit
        #     raise NgSpiceCircuitError('')

        ####################################################################################################

        import math

        import matplotlib.pyplot as plt

        ####################################################################################################

        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Spice.Netlist import Circuit
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared

        ####################################################################################################

        class MyNgSpiceShared(NgSpiceShared):

            ##############################################

            def __init__(self, amplitude, frequency, **kwargs):
                super().__init__(**kwargs)

                self._amplitude = amplitude
                self._pulsation = float(frequency.pulsation)

            ##############################################

            def get_vsrc_data(self, voltage, time, node, ngspice_id):
                self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
                voltage[0] = self._amplitude * math.sin(self._pulsation * time)
                return 0

            ##############################################

            def get_isrc_data(self, current, time, node, ngspice_id):
                self._logger.debug('ngspice_id-{} get_isrc_data @{} node {}'.format(ngspice_id, time, node))
                current[0] = 1.
                return 0

        ####################################################################################################

        circuit = Circuit('Voltage Divider')

        circuit.V('input', 'input', circuit.gnd, 'dc 0 external')
        circuit.R(1, 'input', 'output', 10)
        circuit.R(2, 'output', circuit.gnd, 1)

        amplitude = 10
        frequency = 50
        ngspice_shared = MyNgSpiceShared(amplitude=amplitude, frequency=frequency, send_data=False)
        simulator = circuit.simulator(temperature=25, nominal_temperature=25,
                                      simulator='ngspice-shared', ngspice_shared=ngspice_shared)
        period = float(frequency.period)
        analysis = simulator.transient(step_time=period / 200, end_time=period * 2)

        ####################################################################################################

        figure1, ax = plt.subplots(figsize=(20, 10))
        ax.set_title('Voltage Divider')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [V]')
        ax.grid()
        ax.plot(analysis.input)
        ax.plot(analysis.output)
        ax.legend(('input', 'output'), loc=(.05, .1))
        ax.set_ylim(float(-amplitude * 1.1), float(amplitude * 1.1))

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure1', 'voltage-divider.png')

    def test_ngspice_interpreter(self):
        print("test_ngspice_interpreter")

        ####################################################################################################

        # r#
        # r# =====================
        # r#  NgSpice Interpreter
        # r# =====================
        # r#
        # r# This example explains how to use the NgSpice binding.
        # r#

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Spice.NgSpice.Shared import NgSpiceShared

        ####################################################################################################

        ngspice = NgSpiceShared.new_instance()

        print(ngspice.exec_command('version -f'))
        print(ngspice.exec_command('print all'))
        print(ngspice.exec_command('devhelp'))
        print(ngspice.exec_command('devhelp resistor'))

        circuit = '''
        .title Voltage Multiplier

        .SUBCKT 1N4148 1 2
        *
        R1 1 2 5.827E+9
        D1 1 2 1N4148
        *
        .MODEL 1N4148 D
        + IS = 4.352E-9
        + N = 1.906
        + BV = 110
        + IBV = 0.0001
        + RS = 0.6458
        + CJO = 7.048E-13
        + VJ = 0.869
        + M = 0.03
        + FC = 0.5
        + TT = 3.48E-9
        .ENDS

        Vinput in 0 DC 0V AC 1V SIN(0V 10V 50Hz 0s 0Hz)
        C0 in 1 1mF
        X0 1 0 1N4148
        C1 0 2 1mF
        X1 2 1 1N4148
        C2 1 3 1mF
        X2 3 2 1N4148
        C3 2 4 1mF
        X3 4 3 1N4148
        C4 3 5 1mF
        X4 5 4 1N4148
        R1 5 6 1MegOhm
        .options TEMP = 25°C
        .options TNOM = 25°C
        .options filetype = binary
        .options NOINIT
        .ic
        .tran 0.0001s 0.4s 0s
        .end
        '''

        ngspice.load_circuit(circuit)
        print('Loaded circuit:')
        print(ngspice.listing())

        print(ngspice.show('c3'))
        print(ngspice.showmod('c3'))

        ngspice.run()
        print('Plots:', ngspice.plot_names)

        print(ngspice.ressource_usage())
        print(ngspice.status())

        plot = ngspice.plot(simulation=None, plot_name=ngspice.last_plot)
        print(plot)

        # ngspice.quit()

    def test_operational_amplifier_v1(self):
        print("test_operationalamplifier")

        ####################################################################################################

        from PySpice.Spice.Netlist import SubCircuitFactory

        ####################################################################################################

        class BasicOperationalAmplifier(SubCircuitFactory):
            NAME = 'BasicOperationalAmplifier'
            NODES = ('non_inverting_input', 'inverting_input', 'output')

            ##############################################

            def __init__(self):
                super().__init__()

                # Input impedance
                self.R('input', 'non_inverting_input', 'inverting_input', 10)

                # dc gain=100k and pole1=100hz
                # unity gain = dcgain x pole1 = 10MHZ
                self.VCVS('gain', 1, self.gnd, 'non_inverting_input', 'inverting_input', voltage_gain=100*10^(3))
                self.R('P1', 1, 2, 1)
                self.C('P1', 2, self.gnd, 1.5915)

                # Output buffer and resistance
                self.VCVS('buffer', 3, self.gnd, 2, self.gnd, 1)
                self.R('out', 3, 'output', 10)

        ####################################################################################################

        class BasicComparator(SubCircuitFactory):
            NAME = 'BasicComparator'
            NODES = ('non_inverting_input', 'inverting_input',
                     'voltage_plus', 'voltage_minus',
                     'output')

            ##############################################

            def __init__(self, ):
                super().__init__()

                # Fixme: ngspice is buggy with such subcircuit

                # Fixme: how to pass voltage_plus, voltage_minus ?
                # output_voltage_minus, output_voltage_plus = 0, 15

                # to plug the voltage source
                self.R(1, 'voltage_plus', 'voltage_minus', 1)
                self.NonLinearVoltageSource(1, 'output', 'voltage_minus',
                                            expression='V(non_inverting_input, inverting_input)',
                                            # table=((-micro(1), output_voltage_minus),
                                            #       (micro(1), output_voltage_plus))
                                            table=(('-1uV', '0V'), ('1uV', '15V'))
                                            )

    def test_operational_amplifier_v2(self):
        print("test_operational_amplifier")

        ####################################################################################################

        import numpy as np

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Plot.BodeDiagram import bode_diagram
        from PySpice.Spice.Netlist import Circuit

        # from OperationalAmplifier import BasicOperationalAmplifier

        # f# literal_include('OperationalAmplifier.py')

        ####################################################################################################

        circuit = Circuit('Operational Amplifier')

        # AC 1 PWL(0US 0V  0.01US 1V)
        circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd, amplitude=1)
        # circuit.subcircuit(BasicOperationalAmplifier())
        circuit.X('op', 'BasicOperationalAmplifier', 'in', circuit.gnd, 'out')
        circuit.R('load', 'out', circuit.gnd, 470)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.ac(start_frequency=1, stop_frequency=100, number_of_points=5,
                                variation='dec')

        figure, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

        plt.title("Bode Diagram of an Operational Amplifier")
        bode_diagram(axes=(ax1, ax2),
                     frequency=analysis.frequency,
                     gain=20 * np.log10(np.absolute(analysis.out)),
                     phase=np.angle(analysis.out, deg=False),
                     marker='.',
                     color='blue',
                     linestyle='-',
                     )
        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'operational-amplifier.png')

    def test_capacitor_inductor(self):
        print("test_capacitor_inductor")

        # r# This example shows the simulation of a capacitor and an inductor.
        # r#
        # r# To go further, you can read these pages on Wikipedia: `RC circuit <https://en.wikipedia.org/wiki/RC_circuit>`_
        # r# and `RL circuit <https://en.wikipedia.org/wiki/RL_circuit>`_.

        ####################################################################################################

        import numpy as np
        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Spice.Netlist import Circuit

        from scipy.optimize import curve_fit

        ####################################################################################################

        # Warning: the capacitor/inductor return current in the generator
        #  could use switches instead

        # r# We will use a simple circuit where both capacitor and inductor are driven by a pulse source
        # r# through a limiting current resistor.

        # f# circuit_macros('capacitor_and_inductor.m4')

        # Fixme: for loop makes difficult to intermix code and text !

        # r# We will fit from the simulation output the time constant of each circuit and compare it to the
        # r# theoretical value.

        figure, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

        element_types = ('capacitor', 'inductor')

        for element_type in ('capacitor', 'inductor'):

            circuit = Circuit(element_type.title())
            # Fixme: compute value
            source = circuit.PulseVoltageSource('input', 'in', circuit.gnd,
                                                initial_value=0, pulsed_value=10,
                                                pulse_width=10, period=20)
            circuit.R(1, 'in', 'out', 1)

            if element_type == 'capacitor':
                element = circuit.C
                value = 1
                # tau = RC = 1 ms
            else:
                element = circuit.L
                # Fixme: force component value to an Unit instance ?
                value = 1
                # tau = L/R = 1 ms

            element(1, 'out', circuit.gnd, value)
            # circuit.R(2, 'out', circuit.gnd, kilo(1)) # for debug

            if element_type == 'capacitor':
                tau = circuit['R1'].resistance * circuit['C1'].capacitance
            else:
                tau = circuit['L1'].inductance / circuit['R1'].resistance

            simulator = circuit.simulator(temperature=25, nominal_temperature=25)
            step_time = 10
            analysis = simulator.transient(step_time=step_time, end_time=source.period * 3)

            # Let define the theoretical output voltage.
            if element_type == 'capacitor':
                def out_voltage(t, tau):
                    # Fixme: TypeError: only length-1 arrays can be converted to Python scalars
                    return float(source.pulsed_value) * (1 - np.exp(-t / tau))
            else:
                def out_voltage(t, tau):
                    return float(source.pulsed_value) * np.exp(-t / tau)

            # Fixme: get step_time from analysis
            # At t = 5 tau, each circuit has nearly reached it steady state.
            i_max = int(5 * tau / float(step_time))
            popt, pcov = curve_fit(out_voltage, analysis.out.abscissa[:i_max], analysis.out[:i_max])
            tau_measured = popt[0]

            # Fixme: use Unit().canonise()
            print('tau {0} = {1}'.format(element_type, tau.canonise().str_space()))
            print('tau measured {0} = {1:.1f} ms'.format(element_type, tau_measured * 1000))

            if element_type == 'capacitor':
                ax = ax1
                title = "Capacitor: voltage is constant"
            else:
                ax = ax2
                title = "Inductor: current is constant"
            ax.set_title(title)
            ax.grid()
            current_scale = 1000
            ax.plot(analysis['in'])
            ax.plot(analysis['out'])

            # Fixme: resistor current, scale
            ax.plot(((analysis['in'] - analysis.out) / circuit['R1'].resistance) * current_scale)
            ax.axvline(x=float(tau), color='red')
            ax.set_ylim(-11, 11)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('[V]')
            ax.legend(('Vin [V]', 'Vout [V]', 'I'), loc=(.8, .8))
        # o#

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'capacitor-inductor.png')

        # Fixme: Add formulae

    def test_hp54501a(self):
        print("test_hp54501a")

        ####################################################################################################

        from PySpice.Spice.Netlist import SubCircuitFactory

        ####################################################################################################

        class HP54501A(SubCircuitFactory):
            NAME = 'HP54501A'
            NODES = ('line_plus', 'line_minus')

            ##############################################

            def __init__(self, diode_model):
                super().__init__()

                self.C(1, 'line_plus', 'line_minus', 1)

                self.X('D1', diode_model, 'top', 'line_plus')
                self.X('D2', diode_model, 'line_plus', 'scope_ground')
                self.X('D3', diode_model, 'top', 'line_minus')
                self.X('D4', diode_model, 'line_minus', 'scope_ground')

                self.R(1, 'top', 'output', 10)
                self.C(2, 'output', 'scope_ground', 50)
                self.R(2, 'output', 'scope_ground', 900)
    
    def test_capacitive_half_wave_rectification_post_zener(self):
        print("test_capacitive_half_wave_rectification_post_zener")

        # r# This example shows a capacitive power supply with a post zener half-rectification, a kind
        # r# of transformless power supply.

        # r# To go further on this topic, you can read these design notes:
        # r#
        # r# * Transformerless Power Supply Design, Designer Circuits, LLC
        # r# * Low-cost power supply for home appliances, STM, AN1476
        # r# * Transformerless Power Supplies: Resistive and Capacitive, Microchip, AN954

        ####################################################################################################

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Doc.ExampleTools import find_libraries
        from PySpice.Spice.Library import SpiceLibrary
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        # libraries_path = find_libraries()
        # spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        # f# circuit_macros('capacitive-half-wave-rectification-post-zener-circuit.m4')

        circuit = Circuit('Capacitive Half-Wave Rectification (Post Zener)')

        # circuit.include(spice_library['1N4148'])
        # 1N5919B: 5.6 V, 3.0 W Zener Diode Voltage Regulator
        # circuit.include(spice_library['d1n5919brl'])

        ac_line = circuit.AcLine('input', 'L', circuit.gnd, rms_voltage=230, frequency=50)
        circuit.R('in', 'L', 1, 470)
        circuit.C('in', 1, 2, 470)
        # d1n5919brl pinning is anode cathode ->|-
        # circuit.X('Dz', 'd1n5919brl', circuit.gnd, 2)
        # 1N4148 pinning is anode cathode ->|-
        # circuit.X('D', '1N4148', 2, 'out')
        circuit.C('', 'out', circuit.gnd, 220)
        circuit.R('load', 'out', circuit.gnd, 1)

        # ?# Fixme: circuit.nodes[2].v, circuit.branch.current
        # print circuit.nodes

        # Simulator(circuit, ...).transient(...)
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=ac_line.period / 200, end_time=ac_line.period * 10)

        figure, ax = plt.subplots(figsize=(20, 10))

        ax.plot(analysis['L'] / 100)
        ax.plot(analysis.out)
        ###ax.plot((analysis.out - analysis['L']) / 100)
        ###ax.plot(analysis.out - analysis['2'])
        ###ax.plot((analysis['2'] - analysis['1']) / 100)
        # or:
        #   plt.ax.plot(analysis.out.abscissa, analysis.out)
        ax.legend(('Vin [V]', 'Vout [V]'), loc=(.8, .8))
        ax.grid()
        ax.set_xlabel('t [s]')
        ax.set_ylabel('[V]')

        plt.tight_layout()
        plt.show()
        # f# save_figure('figure', 'capacitive-half-wave-rectification-post-zener.png')

    def test_capacitive_half_wave_rectification_pre_zener(self):
        print("test_capacitive_half_wave_rectification_pre_zener")

        # r# This example shows a capacitive power supply with a pre zener half-rectification.

        ####################################################################################################

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Doc.ExampleTools import find_libraries
        from PySpice.Spice.Library import SpiceLibrary
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        # f# circuit_macros('capacitive-half-wave-rectification-pre-zener-circuit.m4')

        circuit = Circuit('Capacitive Half-Wave Rectification (Pre Zener)')

        circuit.include(spice_library['1N4148'])
        # 1N5919B: 5.6 V, 3.0 W Zener Diode Voltage Regulator
        circuit.include(spice_library['d1n5919brl'])

        ac_line = circuit.AcLine('input', 'L', circuit.gnd, rms_voltage=230, frequency=50)
        circuit.C('in', 'L', 1, 330)
        circuit.R('emi', 'L', 1, 165)
        circuit.R('in', 1, 2, 2 * 47)
        # 1N4148 pinning is anode cathode ->|-
        circuit.X('D1', '1N4148', 2, 'out')
        circuit.C('2', 'out', 3, 250)
        circuit.R('2', 3, circuit.gnd, 1)
        circuit.X('D2', '1N4148', 3, 2)
        # d1n5919brl pinning is anode cathode ->|-
        circuit.X('Dz', 'd1n5919brl', circuit.gnd, 'out')
        circuit.C('', 'out', circuit.gnd, 250)
        circuit.R('load', 'out', circuit.gnd, 1)

        # ?# Fixme: circuit.nodes[2].v, circuit.branch.current
        # print circuit.nodes

        # Simulator(circuit, ...).transient(...)
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=ac_line.period / 200, end_time=ac_line.period * 50)

        figure, ax = plt.subplots(1, figsize=(20, 10))

        ax.plot(analysis.L / 100)
        ax.plot(analysis.out)

        ax.plot(analysis['2'])
        ax.plot(analysis['3'])
        ax.plot((analysis.out - analysis['3']))
        # ax.plot((analysis['2'] - analysis['3']))

        # ax.plot((analysis.L - analysis['1']) / 100)

        ###ax.plot((analysis.out - analysis['L']) / 100)
        ###ax.plot(analysis.out - analysis['2'])
        ###ax.plot((analysis['2'] - analysis['1']) / 100)
        # or:
        #   plt.ax.plot(analysis.out.abscissa, analysis.out)
        ax.legend(('Vin [V]', 'Vout [V]', 'V2 [V]', 'V3 [V]', 'VC2 [V]'), loc=(.8, .8))
        ax.grid()
        ax.set_xlabel('t [s]')
        ax.set_ylabel('[V]')

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'capacitive-half-wave-rectification-post-zener.png')

    def test_cem_simulation(self):
        print("test_cem_simulation")

        # r# ================
        # r#  CEM Simulation
        # r# ================

        # r# This example show a CEM simulation.

        # Fixme: retrieve PDF reference and complete

        ####################################################################################################

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Doc.ExampleTools import find_libraries
        from PySpice.Spice.Library import SpiceLibrary
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        # from HP54501A import HP54501A

        # f# literal_include('HP54501A.py')

        ####################################################################################################

        circuit = Circuit('HP54501A CEM')
        circuit.include(spice_library['1N4148'])
        diode_model = '1N4148'
        ac_line = circuit.AcLine('input', 'input', circuit.gnd, rms_voltage=230, frequency=50)
        # circuit.subcircuit(HP54501A(diode_model='1N4148'))
        # circuit.X('hp54501a', 'HP54501A', 'input', circuit.gnd)
        circuit.C(1, 'input', circuit.gnd, 1)
        circuit.X('D1', diode_model, 'line_plus', 'top')
        circuit.X('D2', diode_model, 'scope_ground', 'input')
        circuit.X('D3', diode_model, circuit.gnd, 'top')
        circuit.X('D4', diode_model, 'scope_ground', circuit.gnd)
        circuit.R(1, 'top', 'output', 10)
        circuit.C(2, 'output', 'scope_ground', 50)
        circuit.R(2, 'output', 'scope_ground', 900)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=ac_line.period / 100, end_time=ac_line.period * 3)

        figure, ax = plt.subplots(figsize=(20, 6))

        ax.plot(analysis.input)
        ax.plot(analysis.Vinput)
        ax.plot(analysis.output - analysis.scope_ground)
        ax.legend(('Vin [V]', 'I [A]'), loc=(.8, .8))
        ax.grid()
        ax.set_xlabel('t [s]')
        ax.set_ylabel('[V]')

        plt.show()

        # f# save_figure('figure', 'hp54501a-cem.png')

    def test_relay_drived_by_a_bipolar_transistor(self):
        print("test_relay_drived_by_a_bipolar_transistor")

        # r# =====================================
        # r#  Relay drived by a bipolar transistor
        # r# =====================================

        # r# This example shows the simulation of ...

        ####################################################################################################

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Doc.ExampleTools import find_libraries
        from PySpice.Spice.Library import SpiceLibrary
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        # ?# #cm# relay.m4

        period = 50
        pulse_width = period / 2

        circuit = Circuit('Relay')

        # circuit.V('digital', 'Vdigital', circuit.gnd, 5@u_V)
        circuit.PulseVoltageSource('clock', 'clock', circuit.gnd, 0, 5, pulse_width, period,
                                   rise_time=5, fall_time=5)
        circuit.R('base', 'clock', 'base', 100)
        circuit.BJT(1, 'collector', 'base', circuit.gnd, model='bjt')  # Q is mapped to BJT !
        circuit.model('bjt', 'npn', bf=80, cjc=5*10^(-12), rb=100)
        circuit.V('analog', 'VccAnalog', circuit.gnd, 8)
        circuit.R('relay', 'VccAnalog', 1, 50)
        circuit.L('relay', 1, 'collector', 100)
        circuit.include(spice_library['1N5822'])  # Schottky diode
        diode = circuit.X('D', '1N5822', 'collector', 'VccAnalog')
        # Fixme: subcircuit node
        # diode.minus.add_current_probe(circuit)

        ####################################################################################################

        figure, ax = plt.subplots(figsize=(20, 10))

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=period / 1000, end_time=period * 1.1)

        ax.set_title('')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [V]')
        ax.grid()
        ax.plot(analysis.base)
        ax.plot(analysis.collector)
        # Fixme: current probe
        # ax.plot((analysis['1'] - analysis.collector)/circuit.Rrelay.resistance)
        ax.plot(analysis['1'] - analysis.collector)
        ax.legend(('Vbase', 'Vcollector'), loc=(.05, .1))

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'relay.png')

    def test_resistor_bridge(self):
        print("test_resistor_bridge")

        # r# This example shows the computation of the DC biases in a resistor bridge.

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        # f# circuit_macros('resistor-bridge.m4')

        circuit = Circuit('Resistor Bridge')

        circuit.V('input', 1, circuit.gnd, 10)
        circuit.R(1, 1, 2, 2)
        circuit.R(2, 1, 3, 1)
        circuit.R(3, 2, circuit.gnd, 1)
        circuit.R(4, 3, circuit.gnd, 2)
        circuit.R(5, 3, 2, 2)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()

        for node in analysis.nodes.values():
            print('Node {}: {:4.1f} V'.format(str(node), float(node)))  # Fixme: format value + unit
        # o#

    def test_voltage_divider(self):
        print('test_voltage_divider')

        # r# This example shows the computation of the DC bias and sensitivity in a voltage divider.

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        # f# circuit_macros('voltage-divider.m4')

        circuit = Circuit('Voltage Divider')

        circuit.V('input', 'in', circuit.gnd, 10)
        circuit.R(1, 'in', 'out', 9)
        circuit.R(2, 'out', circuit.gnd, 1)

        ####################################################################################################

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)

        analysis = simulator.operating_point()
        for node in (analysis['in'], analysis.out):  # .in is invalid !
            print('Node {}: {} V'.format(str(node), float(node)))
        # o#

        # Fixme: Xyce sensitivity analysis
        analysis = simulator.dc_sensitivity('v(out)')
        for element in analysis.elements.values():
            print(element, float(element))
        # o#

    def test_spice_netlist_parser_bootstrap_example(self):
        print("test_spice_netlist_parser_bootstrap_example")

        ####################################################################################################

        # r#
        # r# ========================================
        # r#  Spice Netlist Parser Bootstrap Example
        # r# ========================================
        # r#
        # r# This example shows a bootstrap of a netlist, i.e. we parse the netlist generated by PySpice
        # r# and we regenerate it.
        # r#

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Doc.ExampleTools import find_libraries
        from PySpice.Spice.Library import SpiceLibrary
        from PySpice.Spice.Netlist import Circuit
        from PySpice.Spice.Parser import SpiceParser

        ####################################################################################################

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        circuit = Circuit('STM AN1476: Low-Cost Power Supply For Home Appliances')

        circuit.include(spice_library['1N4148'])
        # 1N5919B: 5.6 V, 3.0 W Zener Diode Voltage Regulator
        circuit.include(spice_library['d1n5919brl'])

        ac_line = circuit.AcLine('input', 'out', 'in', rms_voltage=230, frequency=50)
        circuit.R('load', 'out', circuit.gnd, 1*10^(3))
        circuit.C('load', 'out', circuit.gnd, 220*10^(-6))
        circuit.X('D1', '1N4148', circuit.gnd, 1)
        circuit.D(1, circuit.gnd, 1, model='DIODE1', off=True)
        circuit.X('Dz1', 'd1n5919brl', 1, 'out')
        circuit.C('ac', 1, 2, 470*10^(-9))
        circuit.R('ac', 2, 'in', 470)  # Fixme: , m=1, temperature='{25}'

        source = str(circuit)
        print(source)

        ####################################################################################################

        parser = SpiceParser(source=source)
        bootstrap_circuit = parser.build_circuit()

        bootstrap_source = str(bootstrap_circuit)
        print(bootstrap_source)

        assert (source == bootstrap_source)

    def test_kicad_netlist_parser_example(self):
        print("test_kicad_netlist_parser_example")

        # r#
        # r# ==============================
        # r#  Kicad Netlist Parser Example
        # r# ==============================
        # r#
        # r# This example shows how to read a netlist generated from the |Kicad|_ Schematic Editor.
        # r#
        # r# This example is copied from Stafford Horne's Blog:
        # r#  * http://stffrdhrn.github.io/electronics/2015/04/28/simulating_kicad_schematics_in_spice.html
        # r#  * https://github.com/stffrdhrn/kicad-spice-demo
        # r#
        # r# .. note:: The netlist must be generated using numbered node. Subcircuit elements must have a
        # r#           reference starting by *X* and a value corresponding to the subcircuit's name.
        # r#

        # f# image('kicad-pyspice-example/kicad-pyspice-example.sch.svg')

        # r# The netlist generated by Kicad is the following:

        # f# getthecode('kicad-pyspice-example/kicad-pyspice-example.cir')

        ####################################################################################################

        from pathlib import Path

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Doc.ExampleTools import find_libraries
        from PySpice.Spice.Library import SpiceLibrary
        from PySpice.Spice.Netlist import SubCircuitFactory
        from PySpice.Spice.Parser import SpiceParser

        ####################################################################################################

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        # r# We implement the *PowerIn*, *Opamp*, *JackIn* and *JackOut* elements as subcircuit.

        class PowerIn(SubCircuitFactory):
            NAME = 'PowerIn'
            NODES = ('output_plus', 'ground', 'output_minus')

            ##############################################

            def __init__(self):
                super().__init__()

                self.V('positive', 'output_plus', 'ground', 3.3)
                self.V('negative', 'ground', 'output_minus', 3.3)

        ####################################################################################################

        class Opamp(SubCircuitFactory):
            NAME = 'Opamp'
            NODES = ('output',
                     'input_negative', 'input_positive',
                     'power_positive', 'power_negative')

            ##############################################

            def __init__(self):
                super().__init__()

                self.X('opamp', 'LMV981',
                       'input_positive', 'input_negative',
                       'power_positive', 'power_negative',
                       'output',
                       'NSD')

        ####################################################################################################

        class JackIn(SubCircuitFactory):
            NAME = 'JackIn'
            NODES = ('input', 'x', 'ground')

            ##############################################

            def __init__(self):
                super().__init__()

                # could use SinusoidalVoltageSource as well
                self.V('micro', 'ground', 'input', 'DC 0V AC 1V SIN(0 0.02 440)')

        ####################################################################################################

        class JackOut(SubCircuitFactory):
            NAME = 'JackOut'
            NODES = ('output', 'x', 'ground')

            ##############################################

            def __init__(self):
                super().__init__()

                self.R('load', 'output', 'x', 10)

        ####################################################################################################

        # r# We read the generated netlist.
        directory_path = Path(__file__).resolve().parent
        kicad_netlist_path = directory_path.joinpath('kicad-pyspice-example', 'kicad-pyspice-example.cir')
        parser = SpiceParser(path=str(kicad_netlist_path))

        # r# We build the circuit and translate the ground (5 to 0).
        circuit = parser.build_circuit(ground=5)

        # r# We include the operational amplifier module.
        circuit.include(spice_library['LMV981'])

        # r# We define the subcircuits.
        for subcircuit in (PowerIn(), Opamp(), JackIn(), JackOut()):
            circuit.subcircuit(subcircuit)

        # print(str(circuit))

        # r# We perform a transient simulation.
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=100, end_time=3)

        figure, ax = plt.subplots(figsize=(20, 10))
        ax.plot(analysis['2'])  # JackIn input
        ax.plot(analysis['7'])  # Opamp output
        ax.legend(('Vin [V]', 'Vout [V]'), loc=(.8, .8))
        ax.grid()
        ax.set_xlabel('t [s]')
        ax.set_ylabel('[V]')

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'kicad-example.png')

    def test_buck_converter(self):
        print("test_buck_converter")

        ####################################################################################################

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Doc.ExampleTools import find_libraries
        from PySpice.Probe.Plot import plot
        from PySpice.Spice.Library import SpiceLibrary
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        # ?# circuit_macros('buck-converter.m4')

        circuit = Circuit('Buck Converter')

        circuit.include(spice_library['1N5822'])  # Schottky diode
        circuit.include(spice_library['irf150'])

        # From Microchip WebSeminars - Buck Converter Design Example

        Vin = 12
        Vout = 5
        ratio = Vout / Vin

        Iload = 2
        Rload = Vout / (.8 * Iload)

        frequency = 400
        period = frequency.period
        duty_cycle = ratio * period

        ripple_current = .3 * Iload  # typically 30 %
        ripple_voltage = 50

        print('ratio =', ratio)
        print('RLoad =', Rload)
        print('period =', period.canonise())
        print('duty_cycle =', duty_cycle.canonise())
        print('ripple_current =', ripple_current)

        # r# .. math:
        # r#      U = L \frac{dI}{dt}

        L = (Vin - Vout) * duty_cycle / ripple_current
        RL = 37

        # r# .. math:
        # r#      dV = dI (ESR + \frac{dt}{C} + \frac{ESL}{dt})

        ESR = 30
        ESL = 0
        Cout = (ripple_current * duty_cycle) / (ripple_voltage - ripple_current * ESR)

        ripple_current_in = Iload / 2
        ripple_voltage_in = 200
        ESR_in = 120
        Cin = duty_cycle / (ripple_voltage_in / ripple_current_in - ESR_in)

        L = L.canonise()
        Cout = Cout.canonise()
        Cin = Cin.canonise()

        print('L =', L)
        print('Cout =', Cout)
        print('Cint =', Cin)

        circuit.V('in', 'in', circuit.gnd, Vin)
        circuit.C('in', 'in', circuit.gnd, Cin)

        # Fixme: out drop from 12V to 4V
        # circuit.VCS('switch', 'gate', circuit.gnd, 'in', 'source', model='Switch', initial_state='off')
        # circuit.PulseVoltageSource('pulse', 'gate', circuit.gnd, 0@u_V, Vin, duty_cycle, period)
        # circuit.model('Switch', 'SW', ron=1@u_mΩ, roff=10@u_MΩ)

        # Fixme: Vgate => Vout ???
        circuit.X('Q', 'irf150', 'in', 'gate', 'source')
        # circuit.PulseVoltageSource('pulse', 'gate', 'source', 0@u_V, Vin, duty_cycle, period)
        circuit.R('gate', 'gate', 'clock', 1)
        circuit.PulseVoltageSource('pulse', 'clock', circuit.gnd, 0, 2. * Vin, duty_cycle, period)

        circuit.X('D', '1N5822', circuit.gnd, 'source')
        circuit.L(1, 'source', 1, L)
        circuit.R('L', 1, 'out', RL)
        circuit.C(1, 'out', circuit.gnd, Cout)  # , initial_condition=0@u_V
        circuit.R('load', 'out', circuit.gnd, Rload)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=period / 300, end_time=period * 150)

        figure, ax = plt.subplots(figsize=(20, 10))

        ax.plot(analysis.out)
        ax.plot(analysis['source'])
        # ax.plot(analysis['source'] - analysis['out'])
        # ax.plot(analysis['gate'])
        ax.axhline(y=float(Vout), color='red')
        ax.legend(('Vout [V]', 'Vsource [V]'), loc=(.8, .8))
        ax.grid()
        ax.set_xlabel('t [s]')
        ax.set_ylabel('[V]')

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'buck-converter.png')

    def test_transformer_v1(self):
        print("test_transformer")

        ####################################################################################################

        from PySpice.Spice.Netlist import SubCircuitFactory

        ####################################################################################################

        class Transformer(SubCircuitFactory):

            NAME = 'Transformer'
            NODES = ('input_plus', 'input_minus',
                     'output_plus', 'output_minus')

            ##############################################

            def __init__(self,
                         turn_ratio,
                         primary_inductance=1,
                         copper_resistance=1,
                         leakage_inductance=1,
                         winding_capacitance=20,
                         coupling=0.999,
                         ):

                super().__init__()

                # For an ideal transformer you can reduce the values for the flux leakage inductances, the
                # copper resistors and the winding capacitances. But
                if copper_resistance <= 0:
                    raise ValueError("copper resistance must be > 0")
                if leakage_inductance <= 0:
                    raise ValueError("leakage inductance must be > 0")

                # primary_turns =
                # secondary_turns =
                # turn_ratio = primary_turns / secondary_turns
                # primary_inductance =
                # primary_inductance / secondary_inductance = turn_ratio**2
                secondary_inductance = primary_inductance / float(turn_ratio ** 2)

                # Primary
                self.C('primary', 'input_plus', 'input_minus', winding_capacitance)
                self.L('primary_leakage', 'input_plus', 1, leakage_inductance)
                primary_inductor = self.L('primary', 1, 2, primary_inductance)
                self.R('primary', 2, 'output_minus', copper_resistance)

                # Secondary
                self.C('secondary', 'output_plus', 'output_minus', winding_capacitance)
                self.L('secondary_leakage', 'output_plus', 3, leakage_inductance)
                secondary_inductor = self.L('secondary', 3, 4, secondary_inductance)
                self.R('secondary', 4, 'output_minus', copper_resistance)

                # Coupling
                self.CoupledInductor('coupling', primary_inductor.name, secondary_inductor.name, coupling)

    def test_transformer_v2(self):
        print('test_transformer_v2')

        ####################################################################################################

        # r#
        # r# =============
        # r#  Transformer
        # r# =============
        # r#
        # r# This examples shows how to simulate a transformer.
        # r#

        ####################################################################################################

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Probe.Plot import plot
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        # from Transformer import Transformer

        # f# literal_include('Transformer.py')

        ####################################################################################################

        circuit = Circuit('Transformer')

        ac_line = circuit.AcLine('input', 'input', circuit.gnd, rms_voltage=230, frequency=50)
        # circuit.subcircuit(Transformer(turn_ratio=10))
        circuit.X('transformer', 'Transformer', 'input', circuit.gnd, 'output', circuit.gnd)
        circuit.R('load', 'output', circuit.gnd, 1)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=ac_line.period / 200, end_time=ac_line.period * 3)

        figure, ax = plt.subplots(figsize=(20, 10))
        ax.plot(analysis.input)
        ax.plot(analysis.output)
        ax.legend(('Vin [V]', 'Vout [V]'), loc=(.8, .8))
        ax.grid()
        ax.set_xlabel('t [s]')
        ax.set_ylabel('[V]')

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'transformer.png')

    def test_ac_coupled_amplifier(self):
        print("test_ac_coupled_amplifier")

        # r# ======================
        # r#  AC Coupled Amplifier
        # r# ======================

        # r# This example shows the simulation of an AC coupled amplifier using a NPN bipolar transistor.

        ####################################################################################################

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Doc.ExampleTools import find_libraries
        from PySpice.Spice.Library import SpiceLibrary
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        # f# circuit_macros('ac-coupled-amplifier.m4')

        circuit = Circuit('Transistor')

        circuit.V('power', 5, circuit.gnd, 15)
        source = circuit.SinusoidalVoltageSource('in', 'in', circuit.gnd, amplitude=0.5, frequency=1)
        circuit.C(1, 'in', 2, 10)
        circuit.R(1, 5, 2, 100)
        circuit.R(2, 2, 0, 20)
        circuit.R('C', 5, 4, 10)
        circuit.BJT(1, 4, 2, 3, model='bjt')  # Q is mapped to BJT !
        circuit.model('bjt', 'npn', bf=80, cjc=5*10^(-12), rb=100)
        circuit.R('E', 3, 0, 2)
        circuit.C(2, 4, 'out', 10)
        circuit.R('Load', 'out', 0, 1)

        ####################################################################################################

        figure, ax = plt.subplots(figsize=(20, 10))

        # .ac dec 5 10m 1G

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=source.period / 200, end_time=source.period * 2)

        ax.set_title('')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage [V]')
        ax.grid()
        ax.plot(analysis['in'])
        ax.plot(analysis.out)
        ax.legend(('input', 'output'), loc=(.05, .1))

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'ac-coupled-amplifier-plot.png')

    def test_n_mosfet_transistor(self):
        print("test_n_mosfet_transistor")

        # r# =====================
        # r#  n-MOSFET Transistor
        # r# =====================

        # r# This example shows how to simulate the characteristic curves of an nmos transistor.

        ####################################################################################################

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Doc.ExampleTools import find_libraries
        from PySpice.Spice.Library import SpiceLibrary
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        # r# We define a basic circuit to drive an nmos transistor using two voltage sources.
        # r# The nmos transistor demonstrated in this example is a low-level device description.

        # ?# TODO: Write the : circuit_macros('nmos_transistor.m4')

        circuit = Circuit('NMOS Transistor')
        circuit.include(spice_library['ptm65nm_nmos'])

        # Define the DC supply voltage value
        Vdd = 1.1

        # Instanciate circuit elements
        Vgate = circuit.V('gate', 'gatenode', circuit.gnd, 0)
        Vdrain = circuit.V('drain', 'vdd', circuit.gnd, Vdd)
        # M <name> <drain node> <gate node> <source node> <bulk/substrate node>
        circuit.MOSFET(1, 'vdd', 'gatenode', circuit.gnd, circuit.gnd, model='ptm65nm_nmos')

        # r# We plot the characteristics :math:`Id = f(Vgs)` using a DC sweep simulation.

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.dc(Vgate=slice(0, Vdd, .01))

        figure, ax = plt.subplots(figsize=(20, 10))

        ax.plot(analysis['gatenode'], -analysis.Vdrain)
        ax.legend('NMOS characteristic')
        ax.grid()
        ax.set_xlabel('Vgs [V]')
        ax.set_ylabel('Id [mA]')

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'transistor-nmos-plot.png')

    def test_bipolar_transistor(self):
        print("test_bipolar_transistor")

        # r# ====================
        # r#  Bipolar Transistor
        # r# ====================

        # r# This example shows how to simulate the characteristic curves of a bipolar transistor.

        # Fixme: Complete

        ####################################################################################################

        import numpy as np
        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Doc.ExampleTools import find_libraries
        from PySpice.Spice.Library import SpiceLibrary
        from PySpice.Spice.Netlist import Circuit

        ####################################################################################################

        libraries_path = find_libraries()
        spice_library = SpiceLibrary(libraries_path)

        ####################################################################################################

        figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))

        ####################################################################################################

        # r# We define a basic circuit to drive an NPN transistor (2n2222a) using two voltage sources.

        # f# circuit_macros('transistor.m4')

        circuit = Circuit('Transistor')

        Vbase = circuit.V('base', '1', circuit.gnd, 1)
        circuit.R('base', 1, 'base', 1)
        Vcollector = circuit.V('collector', '2', circuit.gnd, 0)
        circuit.R('collector', 2, 'collector', 1)
        # circuit.BJT(1, 'collector', 'base', circuit.gnd, model='generic')
        # circuit.model('generic', 'npn')
        circuit.include(spice_library['2n2222a'])
        circuit.BJT(1, 'collector', 'base', circuit.gnd, model='2n2222a')

        # r# We plot the base-emitter diode curve :math:`Ib = f(Vbe)` using a DC sweep simulation.

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.dc(Vbase=slice(0, 3, .01))

        ax1.plot(analysis.base, -analysis.Vbase)  # Fixme: I_Vbase
        ax1.axvline(x=.65, color='red')
        ax1.legend(('Base-Emitter Diode curve',), loc=(.1, .8))
        ax1.grid()
        ax1.set_xlabel('Vbe [V]')
        ax1.set_ylabel('Ib [mA]')

        ####################################################################################################

        # r# We will now replace the base's voltage source by a current source in the previous circuit.

        circuit = Circuit('Transistor')
        Ibase = circuit.I('base', circuit.gnd, 'base', 10)  # take care to the orientation
        Vcollector = circuit.V('collector', 'collector', circuit.gnd, 5)
        # circuit.BJT(1, 'collector', 'base', circuit.gnd, model='generic')
        # circuit.model('generic', 'npn')
        circuit.include(spice_library['2n2222a'])
        circuit.BJT(1, 'collector', 'base', circuit.gnd, model='2n2222a')

        # Fixme: ngspice doesn't support multi-sweep ???
        #   it works in interactive mode

        # ?# simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        # ?# analysis = simulator.dc(Vcollector=slice(0, 5, .1), Ibase=slice(micro(10), micro(100), micro(10)))
        # ?# 0 v(i-sweep)    voltage # Vcollector in fact
        # ?# 1 v(collector)  voltage
        # ?# 2 v(base)       voltage
        # ?# 3 i(vcollector) current
        # ?# 0.00000000e+00,   1.00000000e-01,   2.00000000e-01, 3.00000000e-01,   4.00000000e-01,   5.00000000e-01, 6.00000000e-01,   7.00000000e-01,   8.00000000e-01, 9.00000000e-01
        # ?# 0.00000000e+00,   1.00000000e-01,   2.00000000e-01, 3.00000000e-01,   4.00000000e-01,   5.00000000e-01, 6.00000000e-01,   7.00000000e-01,   8.00000000e-01, 9.00000000e-01
        # ?# 6.50478604e-01,   7.40522920e-01,   7.68606463e-01, 7.69192913e-01,   7.69049191e-01,   7.69050844e-01, 7.69049584e-01,   7.69049559e-01,   7.69049559e-01, 7.69049559e-01
        # ?# 9.90098946e-06,  -3.15540984e-04,  -9.59252614e-04, -9.99134834e-04,  -9.99982226e-04,  -1.00005097e-03, -1.00000095e-03,  -9.99999938e-04,  -9.99999927e-04, -9.99999937e-04
        # ?#
        # ?# analysis = simulator.dc(Vcollector=slice(0, 10, .1))
        # ?# 0 v(v-sweep)      voltage
        # ?# 1 v(collector)    voltage
        # ?# 2 v(base)         voltage
        # ?# 3 i(vcollector)   current
        # ?#
        # ?# analysis = simulator.dc(Ibase=slice(micro(10), micro(100), micro(10)))
        # ?# 0 v(i-sweep)      voltage
        # ?# 1 v(collector)    voltage
        # ?# 2 v(base)         voltage
        # ?# 3 i(vcollector)   current

        ax2.grid()
        # ax2.legend(('Ic(Vce, Ib)',), loc=(.5,.5))
        ax2.set_xlabel('Vce [V]')
        ax2.set_ylabel('Ic [mA]')
        ax2.axvline(x=.2, color='red')

        ax3.grid()
        # ax3.legend(('beta(Vce)',), loc=(.5,.5))
        ax3.set_xlabel('Vce [V]')
        ax3.set_ylabel('beta')
        ax3.axvline(x=.2, color='red')

        for base_current in np.arange(0, 100, 10):
            base_current = base_current
            Ibase.dc_value = base_current
            simulator = circuit.simulator(temperature=25, nominal_temperature=25)
            analysis = simulator.dc(Vcollector=slice(0, 5, .01))
            # add ib as text, linear and saturate region
            # Plot Ic = f(Vce)
            ax2.plot(analysis.collector, -analysis.Vcollector)
            # Plot β = Ic / Ib = f(Vce)
            ax3.plot(analysis.collector, -analysis.Vcollector / float(base_current))
            # trans-resistance U = RI   R = U / I = Vce / Ie
            # ax3.plot(analysis.collector, analysis.sweep/(float(base_current)-analysis.Vcollector))
            # Fixme: sweep is not so explicit

        # r# Let plot :math:`Ic = f(Ib)`

        ax4.grid()
        ax4.set_xlabel('Ib [uA]')
        ax4.set_ylabel('Ic [mA]')

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.dc(Ibase=slice(0, 100e-6, 10e-6))
        # Fixme: sweep
        ax4.plot(analysis.sweep * 1e6, -analysis.Vcollector, 'o-')
        ax4.legend(('Ic(Ib)',), loc=(.1, .8))

        ####################################################################################################

        plt.tight_layout()
        plt.show()

        # f# save_figure('figure', 'transistor-plot.png')

    def test_time_delay(self):
        print("test_time_delay")

        # r# This example shows the simulation of a transmission line.

        ####################################################################################################

        import matplotlib.pyplot as plt

        ####################################################################################################

        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()

        ####################################################################################################

        from PySpice.Probe.Plot import plot
        from PySpice.Spice.Netlist import Circuit
        from PySpice.Unit import *

        ####################################################################################################

        # r# We will drive the transmission line with a pulse source and use a standard 50 Ω load.

        circuit = Circuit('Transmission Line')
        circuit.PulseVoltageSource('pulse', 'input', circuit.gnd, 0, 1, 1, 1)
        circuit.LosslessTransmissionLine('delay', 'output', circuit.gnd, 'input', circuit.gnd,
                                         impedance=50, time_delay=40e-9)
        circuit.R('load', 'output', circuit.gnd, 50)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=1e-11, end_time=100e-9)

        ####################################################################################################

        figure, ax = plt.subplots(figsize=(20, 6))
        ax.plot(analysis['input'])
        ax.plot(analysis['output'])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Voltage (V)')
        ax.grid()
        ax.legend(['input', 'output'], loc='upper right')

        plt.show()

        # f# save_figure('figure', 'time-delay.png')


if __name__ == '__main__':
    unittest.main()
