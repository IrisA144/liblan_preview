from liblan.helper import siso_helper
from pyscf import gto

mol = gto.M(atom='''
                Co    3.204478    5.247118    3.364221
                N    1.734061    6.551627    3.166054
                N    2.692488    3.373408    3.730354
                N    5.055819    5.854666    3.763880
                N    2.895467    5.515551    5.391701
                O    0.659775    8.172282    4.458338
                O    1.319496    2.263397    5.260980
                O    6.524024    5.842837    5.597363
                C    2.383645    6.903557    5.473121
                H    3.130650    7.519682    5.504956
                H    1.874724    7.009038    6.293793
                C    1.487286    7.253517    4.273807
                C    0.882157    6.796105    1.952935
                C    1.464948    5.868467    0.840957
                H    2.367344    6.127732    0.647973
                H    0.931607    5.948317    0.045076
                H    1.450987    4.957588    1.143814
                C   -0.548569    6.417558    2.252412
                H   -0.582122    5.507665    2.558086
                H   -1.076225    6.506280    1.456531
                H   -0.897337    6.996222    2.935601
                C    0.990368    8.231430    1.460757
                H    1.909150    8.439434    1.279043
                H    0.654894    8.826853    2.135495
                H    0.473470    8.333953    0.659242
                C    1.876458    4.495248    5.736536
                H    0.995006    4.889568    5.654272
                H    1.997947    4.228096    6.660039
                C    1.953001    3.238353    4.833881
                C    2.807330    2.207206    2.803471
                C    3.809515    2.627157    1.688957
                H    3.470241    3.399038    1.231149
                H    3.915512    1.904566    1.064930
                H    4.659410    2.836147    2.084784
                C    3.347551    0.989743    3.544132
                H    4.196512    1.204648    3.935734
                H    3.453841    0.262223    2.927149
                H    2.729972    0.737378    4.234365
                C    1.444717    1.867105    2.170711
                H    1.113721    2.632086    1.693183
                H    0.823027    1.626570    2.862352
                H    1.548763    1.131698    1.563588
                C    4.218104    5.316420    5.964454
                H    4.320383    4.378924    6.189554
                H    4.278583    5.825092    6.789634
                C    5.373179    5.726512    5.052502
                C    6.082973    6.265744    2.757831
                C    7.138672    5.180379    2.549071
                H    7.623489    5.044338    3.366644
                H    7.745394    5.452460    1.856584
                H    6.707948    4.361179    2.293262
                C    5.322266    6.478677    1.437092
                H    4.656114    7.158879    1.557953
                H    4.896808    5.658492    1.174804
                H    5.936732    6.751744    0.752213
                C    6.747933    7.574887    3.183521
                H    6.074108    8.248189    3.313116
                H    7.359798    7.859783    2.501740
                H    7.224015    7.438847    4.006165
        ''',
        basis={'default':'def2tzvp','C':'6-31G*','H':'6-31G*','O':'6-31G*'}, symmetry=0 ,spin = 3,charge = -1,verbose= 4)

siso_helper.cassi_so(mol,[0,35,0,10],['Co 3d'],'CoSMM9')