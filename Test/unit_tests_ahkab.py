# ahkab
import os
import unittest
import ahkab
from ahkab import new_ac, run
from ahkab.circuit import Circuit
from ahkab import circuit, time_functions
import pylab as plt
import numpy as np
import scipy, scipy.interpolate
import sympy
from sympy.abc import w
from sympy import I


class UnitTestsElectronicSimulationAhkabExamples(unittest.TestCase):
    # ok
    def test_repository(self):
        print("test_repository")

        # Define the circuit
        cir = Circuit('Butterworth 1kHz band-pass filter')
        cir.add_vsource('V1', 'n1', cir.gnd, dc_value=0., ac_value=1.)
        cir.add_resistor('R1', 'n1', 'n2', 50.)
        cir.add_inductor('L1', 'n2', 'n3', 0.245894)
        cir.add_capacitor('C1', 'n3', 'n4', 1.03013e-07)
        cir.add_inductor('L2', 'n4', cir.gnd, 9.83652e-05)
        cir.add_capacitor('C2', 'n4', cir.gnd, 0.000257513)
        cir.add_inductor('L3', 'n4', 'n5', 0.795775)
        cir.add_capacitor('C3', 'n5', 'n6', 3.1831e-08)
        cir.add_inductor('L4', 'n6', cir.gnd, 9.83652e-05)
        cir.add_capacitor('C4', 'n6', cir.gnd, 0.000257513)
        cir.add_capacitor('C5', 'n7', 'n8', 1.03013e-07)
        cir.add_inductor('L5', 'n6', 'n7', 0.245894)
        cir.add_resistor('R2', 'n8', cir.gnd, 50.)

        # Define the analysis
        ac1 = new_ac(.97e3, 1.03e3, 1e2, x0=None)

        # run it
        res = run(cir, ac1)

        # plot the results
        fig = plt.figure()
        plt.subplot(211)
        plt.semilogx(res['ac']['f'] * 2 * np.pi, np.abs(res['ac']['Vn8']), 'o-')
        plt.ylabel('abs(V(n8)) [V]')
        plt.title(cir.title + " - AC Simulation")
        plt.subplot(212)
        plt.grid(True)
        plt.semilogx(res['ac']['f'] * 2 * np.pi, np.angle(res['ac']['Vn8']), 'o-')
        plt.xlabel('Angular frequency [rad/s]')
        plt.ylabel('arg(V(n4)) [rad]')
        fig.savefig('bpf_transfer_fn.svg')
        plt.show()

    # ok
    def test_command_line_help(self):
        print("test_command_line_help")

        # os.system("ahkab --help")
        os.system("ahkab --version")

    # ok
    def test_a_first_op_example(self):
        print("test_a_first_op_example")

        mycir = ahkab.Circuit('Simple Example Circuit')
        mycir.add_resistor('R1', 'n1', mycir.gnd, value=5)
        mycir.add_vsource('V1', 'n2', 'n1', dc_value=8)
        mycir.add_resistor('R2', 'n2', mycir.gnd, value=2)
        mycir.add_vsource('V2', 'n3', 'n2', dc_value=4)
        mycir.add_resistor('R3', 'n3', mycir.gnd, value=4)
        mycir.add_resistor('R4', 'n3', 'n4', value=1)
        mycir.add_vsource('V3', 'n4', mycir.gnd, dc_value=10)
        mycir.add_resistor('R5', 'n2', 'n4', value=4)
        opa = ahkab.new_op()
        r = ahkab.run(mycir, opa)['op']
        print(r)

    # ok
    def test_ac_and_tran_tutorial(self):
        print("test_ac_and_tran_tutorial")

        mycircuit = circuit.Circuit(title="Butterworth Example circuit")

        gnd = mycircuit.get_ground_node()

        mycircuit.add_resistor("R1", n1="n1", n2="n2", value=600)
        mycircuit.add_inductor("L1", n1="n2", n2="n3", value=15.24e-3)
        mycircuit.add_capacitor("C1", n1="n3", n2=gnd, value=119.37e-9)
        mycircuit.add_inductor("L2", n1="n3", n2="n4", value=61.86e-3)
        mycircuit.add_capacitor("C2", n1="n4", n2=gnd, value=155.12e-9)
        mycircuit.add_resistor("R2", n1="n4", n2=gnd, value=1.2e3)

        voltage_step = time_functions.pulse(v1=0, v2=1, td=500e-9, tr=1e-12, pw=1, tf=1e-12, per=2)

        mycircuit.add_vsource("V1", n1="n1", n2=gnd, dc_value=5, ac_value=1, function=voltage_step)

        print(mycircuit)
        print("\n")

        op_analysis = ahkab.new_op()
        ac_analysis = ahkab.new_ac(start=1e3, stop=1e5, points=100)
        tran_analysis = ahkab.new_tran(tstart=0, tstop=1.2e-3, tstep=1e-6, x0=None)

        r = ahkab.run(mycircuit, an_list=[op_analysis, ac_analysis, tran_analysis])

        print(r)
        print("\n")

        print(r['op'].results)
        print("\n")

        print(r['op'].keys())
        print("\n")

        print(r['op']['VN4'])
        print("\n")

        print("The DC output voltage is %s %s" % (r['op']['VN4'], r['op'].units['VN4']))
        print("\n")

        print(r['ac'])
        print("\n")

        print(r['ac'].keys())
        print("\n")

        fig = plt.figure()
        plt.title(mycircuit.title + " - TRAN Simulation")
        plt.plot(r['tran']['T'], r['tran']['VN1'], label="Input voltage")
        # plt.hold(True)
        plt.plot(r['tran']['T'], r['tran']['VN4'], label="output voltage")
        plt.legend()
        # plt.hold(False)
        plt.grid(True)
        plt.ylim([0, 1.2])
        plt.ylabel('Step response')
        plt.xlabel('Time [s]')
        fig.savefig('tran_plot.svg')

        fig = plt.figure()
        plt.subplot(211)
        plt.semilogx(r['ac']['f'] * 2 * np.pi, np.abs(r['ac']['Vn4']), 'o-')
        plt.ylabel('abs(V(n4)) [V]')
        plt.title(mycircuit.title + " - AC Simulation")
        plt.subplot(212)
        plt.grid(True)
        plt.semilogx(r['ac']['f'] * 2 * np.pi, np.angle(r['ac']['Vn4']), 'o-')
        plt.xlabel('Angular frequency [rad/s]')
        plt.ylabel('arg(V(n4)) [rad]')
        fig.savefig('ac_plot.svg')
        plt.show()

        # Normalize the output to the low frequency value and convert to array
        norm_out = np.abs(r['ac']['Vn4']) / np.abs(r['ac']['Vn4']).max()
        # Convert to dB
        norm_out_db = 20 * np.log10(norm_out)
        # Convert angular frequencies to Hz and convert matrix to array
        frequencies = r['ac']['f']
        # call scipy to interpolate
        norm_out_db_interpolated = scipy.interpolate.interp1d(frequencies, norm_out_db)

        print("Maximum attenuation in the pass band (0-%g Hz) is %g dB" % (2e3, -1.0 * norm_out_db_interpolated(2e3)))
        print("Minimum attenuation in the stop band (%g Hz - Inf) is %g dB" % (6.5e3, -1.0 * norm_out_db_interpolated(6.5e3)))

    # ok
    def test_pole_zero_example(self):
        print("test_pole_zero_example")

        # 1. Describe the circuit with ahkab
        print("We're using ahkab %s" % ahkab.__version__)
        print("\n")

        bpf = ahkab.Circuit('RLC bandpass')
        bpf.add_inductor('L1', 'in', 'n1', 1e-6)
        bpf.add_capacitor('C1', 'n1', 'out', 2.2e-12)
        bpf.add_resistor('R1', 'out', bpf.gnd, 13)

        # We also give V1 an AC value since we wish to run an AC simulation in the following
        bpf.add_vsource('V1', 'in', bpf.gnd, dc_value=1, ac_value=1)

        print(bpf)
        print('\n')

        # 2. PZ analysis
        pza = ahkab.new_pz('V1', ('out', bpf.gnd), x0=None, shift=1e3)
        r = ahkab.run(bpf, pza)['pz']

        print(r.keys())
        print("\n")

        print('Singularities:')
        for x, _ in r:
            print("* %s = %+g %+gj Hz" % (x, np.real(r[x]), np.imag(r[x])))
        print("\n")

        fig = plt.figure()
        # plot o's for zeros and x's for poles
        for x, v in r:
            plt.plot(np.real(v), np.imag(v), 'bo' * (x[0] == 'z') + 'rx' * (x[0] == 'p'))

        # set axis limits and print some thin axes
        xm = 1e6
        plt.xlim(-xm * 10, xm * 10)
        plt.plot(plt.xlim(), [0, 0], 'k', alpha=.5, lw=.5)
        plt.plot([0, 0], plt.ylim(), 'k', alpha=.5, lw=.5)

        # plot the distance from the origin of p0 and p1
        plt.plot([np.real(r['p0']), 0], [np.imag(r['p0']), 0], 'k--', alpha=.5)
        plt.plot([np.real(r['p1']), 0], [np.imag(r['p1']), 0], 'k--', alpha=.5)

        # print the distance between p0 and p1
        plt.plot([np.real(r['p1']), np.real(r['p0'])], [np.imag(r['p1']), np.imag(r['p0'])], 'k-', alpha=.5, lw=.5)

        # label the singularities
        plt.text(np.real(r['p1']), np.imag(r['p1']) * 1.1, '$p_1$', ha='center', fontsize=20)
        plt.text(.4e6, .4e7, '$z_0$', ha='center', fontsize=20)
        plt.text(np.real(r['p0']), np.imag(r['p0']) * 1.2, '$p_0$', ha='center', va='bottom', fontsize=20)
        plt.xlabel('Real [Hz]')
        plt.ylabel('Imag [Hz]')
        plt. title('Singularities')
        fig.savefig('pz_plot.svg')
        plt.show()

        C = 2.2e-12
        L = 1e-6
        f0 = 1 / (2 * np.pi * np.sqrt(L * C))
        print('Resonance frequency from analytic calculations: %g Hz' % f0)
        print("\n")

        alpha = (-r['p0'] - r['p1']) / 2
        a1 = np.real(abs(r['p0'] - r['p1'])) / 2
        f0 = np.sqrt(a1 ** 2 - alpha ** 2)
        f0 = np.real_if_close(f0)
        print('Resonance frequency from PZ analysis: %g Hz' % f0)
        print("\n")

        # 3. AC analysis
        aca = ahkab.new_ac(start=1e8, stop=5e9, points=5e2, x0=None)
        rac = ahkab.run(bpf, aca)['ac']

        sympy.init_printing()

        p0, p1, z0 = sympy.symbols('p0, p1, z0')

        # constant term, can be calculated to be R/L
        k = 13 / 1e-6

        H = 13 / 1e-6 * (I * w + z0 * 6.28) / (I * w + p0 * 6.28) / (I * w + p1 * 6.28)
        Hl = sympy.lambdify(w, H.subs({p0: r['p0'], z0: abs(r['z0']), p1: r['p1']}))

        def dB20(x):
            return 20 * np.log10(x)

        fig = plt.figure()
        plt.semilogx(rac.get_x(), dB20(abs(rac['vout'])), label='TF from AC analysis')
        plt.semilogx(rac.get_x() / (2*np.pi), dB20(abs(Hl(rac.get_x()))), 'o', ms=4, label='TF from PZ analysis')
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('|H(w)| [dB]')
        plt.xlim(4e7, 3e8)
        plt.ylim(-50, 1)
        fig.savefig('ac_plot.svg')
        plt.show()

        # 4. Symbolic analysis
        symba = ahkab.new_symbolic(source='V1')
        rs, tfs = ahkab.run(bpf, symba)['symbolic']
        print(rs)
        print('\n')

        print(tfs)
        print('\n')

        print(tfs['VOUT/V1'])
        print("\n")

        Hs = tfs['VOUT/V1']['gain']
        print(Hs)
        print("\n")

        s, C1, R1, L1 = rs.as_symbols('s C1 R1 L1')
        HS = sympy.lambdify(w, Hs.subs({s: I * w, C1: 2.2e-12, R1: 13., L1: 1e-6}))

        print(np.allclose(dB20(abs(HS(rac.get_x()))), dB20(abs(Hl(rac.get_x()))), atol=1))

        # 5. Conclusions
        fig = plt.figure()
        plt.title('Series RLC passband: TFs compared')
        plt.semilogx(rac.get_x(), dB20(abs(rac['vout'])), label='TF from AC analysis')
        plt.semilogx(rac.get_x() / 2 / np.pi, dB20(abs(Hl(rac.get_x()))), 'o', ms=4, label='TF from PZ analysis')
        plt.semilogx(rac.get_x() / 2 / np.pi, dB20(abs(HS(rac.get_x()))), '-', lw=10, alpha=.2, label='TF from symbolic analysis')
        plt.vlines(1.07297e+08, * plt.gca().get_ylim(), alpha=.4)
        plt.text(7e8 / 2 / np.pi, -45, '$f_d = 107.297\\, \\mathrm{MHz}$', fontsize=20)
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('|H(w)| [dB]')
        plt.xlim(4e7, 3e8)
        plt.ylim(-50, 1)
        fig.savefig('conclusions.svg')
        plt.show()


if __name__ == '__main__':
    unittest.main()
