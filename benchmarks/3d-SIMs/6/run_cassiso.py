from liblan.dmet import ssdmet
from pyscf import gto, scf

mol = gto.M(atom='''
                Fe  13.601739    3.445053   -0.553411
                N   15.622921    3.541656   -1.341798
                N   14.031396    5.311792    0.050266
                N   13.149964    3.177664   -2.526063
                N   14.375991    1.966664    0.591411
                C   16.477456    2.914704   -0.307851
                C   15.734174    1.810272    0.392029
                C   16.189776    0.696152    1.036593
                C   15.073722    0.120408    1.676346
                C   13.982121    0.900984    1.390141
                C   12.591851    0.783344    1.902424
                C   11.700953   -0.171616    1.394951
                C   10.407584   -0.225592    1.890399
                C    9.957244    0.631104    2.878890
                C   10.849657    1.551464    3.388768
                C   12.153673    1.644192    2.926991
                C   12.110787   -1.115504    0.303041
                C    8.542262    0.553600    3.374338
                C   13.105452    2.642056    3.528263
                C   15.899755    4.978248   -1.532041
                C   15.218461    5.785120   -0.456967
                C   15.516244    7.001656    0.093798
                C   14.477787    7.299216    0.990896
                C   13.585891    6.248760    0.957225
                C   12.345727    6.010712    1.724448
                C   12.351999    5.978880    3.124208
                C   11.177891    5.688240    3.814469
                C    9.980275    5.415592    3.169905
                C    9.979080    5.498632    1.789385
                C   11.116351    5.798960    1.063049
                C   13.608348    6.222464    3.917887
                C    8.733838    5.083432    3.946749
                C   11.018494    6.003792   -0.423295
                C   15.598597    2.777688   -2.599899
                C   14.269463    2.950688   -3.285350
                C   13.935264    2.827512   -4.615363
                C   12.541228    2.986672   -4.685110
                C   12.080432    3.217800   -3.405604
                C   10.702884    3.520896   -2.948637
                C    9.872712    4.417728   -3.670164
                C    8.618118    4.740200   -3.174715
                C    8.116321    4.235040   -1.993818
                C    8.921993    3.350664   -1.308368
                C   10.184485    2.974216   -1.760524
                C   10.281337    5.046064   -4.973721
                C    6.771172    4.632248   -1.457483
                C   10.939369    1.983272   -0.923554
                H   17.256356    2.557632   -0.719121
                H   16.731111    3.574872    0.329497
                H   17.081884    0.372296    1.051023
                H   15.073607   -0.664320    2.210275
                H    9.809340   -0.873304    1.536851
                H   10.562580    2.141048    4.074218
                H    8.405737    1.206848    4.052572
                H    7.944739    0.718296    2.652811
                H    8.378860   -0.310016    3.730291
                H   16.838253    5.126336   -1.488749
                H   15.576400    5.250896   -2.383442
                H   16.277740    7.540032   -0.096204
                H   14.401638    8.082560    1.527231
                H   11.198558    5.684088    4.764478
                H    9.168836    5.338088    1.320393
                H    8.065983    5.736680    3.773582
                H    8.412486    4.228120    3.677379
                H    8.929451    5.068208    4.875112
                H   16.289787    3.090472   -3.169905
                H   15.735729    1.857328   -2.407493
                H   14.529437    2.669736   -5.339294
                H   12.012298    2.943768   -5.473979
                H    8.073939    5.338088   -3.677379
                H    8.600694    2.979752   -0.495448
                H    6.877739    5.041912   -0.606082
                H    6.363512    5.248128   -2.053945
                H    6.222968    3.862744   -1.370900
                H    9.684402    5.753288   -5.185368
                H   11.165516    5.390680   -4.899163
                H   10.255183    4.392816   -5.663981
                H   13.872013    7.128984    3.821684
                H   13.447159    6.035624    4.836631
                H   14.302088    5.655024    3.598011
                H   11.416803    1.389536   -1.488749
                H   10.326151    1.490568   -0.392029
                H   10.105966    5.980264   -0.685450
                H   11.394179    6.845264   -0.649374
                H   11.495934    5.313176   -0.868237
                H   11.547883    2.445528   -0.360763
                H   12.686542    3.076632    4.261815
                H   13.884880    2.190872    3.833709
                H   13.347024    3.284232    2.874079
                H   12.798847   -1.687096    0.622918
                H   11.360407   -1.635888    0.040886
                H   12.430222   -0.618648   -0.440131
        ''',
        basis={'default':'def2tzvp','C':'6-31G*','H':'6-31G*'}, symmetry=0 ,spin = 4,charge = -1,verbose= 4)

title = 'FeMes3'
mf = scf.rohf.ROHF(mol).x2c()
mf.chkfile = title+'_rohf.chk'
mf.init_guess = 'chk'
mf.level_shift = .2
mf.max_cycle = 1
mf.kernel()

mydmet = ssdmet.SSDMET(mf,title=title)
mydmet.kernel(mol.search_ao_label('Fe *'),imp_solver='casscf',imp_solver_soc='siso',statelis=[0,0,45,0,5])