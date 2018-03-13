from ClusterHelper import *
import time


def getChunkCluster(F=None):
	if F is None:
		cs = [[(0, [('business', {'products': 71.20049458742383, 'china': 273.600635055619, 'suppliers': 517.9432034048027, 'manufacturers': 372.7722031254309}), ('datum', {'products': 88.42156266293742, 'china': 369.55140012662685, 'suppliers': 532.3156928079802, 'manufacturers': 504.5845991059355}), ('stakeholder', {'products': 90.10216680807028, 'china': 290.6145070696192, 'suppliers': 420.3044185283913, 'manufacturers': 420.3044185283913}), ('supplier', {'products': 53.98773608440659, 'china': 212.6293038974643, 'suppliers': 378.58411268615424, 'manufacturers': 323.6675259573714})]), (1, [('exporter', {'products': 64.77348312931292, 'china': 127.31783024632082, 'suppliers': 1000000, 'manufacturers': 1000000}), ('factor', {'products': 85.41731565750165, 'china': 306.8654818964716, 'suppliers': 551.8474428225785, 'manufacturers': 623.6013512885434}), ('information', {'products': 82.62977006756947, 'china': 332.3243667570349, 'suppliers': 551.7326416695875, 'manufacturers': 561.6302974255407})]), (2, [('directory', {'products': 1000000, 'china': 266.0705095990702, 'suppliers': 1000000, 'manufacturers': 361.4770149406076}), ('manufacture', {'products': 57.69557156019939, 'china': 269.23681419381865, 'suppliers': 1000000, 'manufacturers': 543.0981403290825}), ('manufacturer', {'products': 56.86348513625482, 'china': 268.5127207834308, 'suppliers': 1000000, 'manufacturers': 537.6667369310992})])], [(0, [('activity', {'homepage': 327.4364980706987, 'power': 185.0143980111469, 'procurement': 216.75824734306465, 'products': 45.26944845486219, 'manufacturing': 64.05481486139047, 'data': 167.51721241237584, 'management': 141.81889533654243, 'electronics': 247.5143980111469}), ('area', {'homepage': 299.56356229729846, 'power': 173.67353485263527, 'procurement': 192.48790958163409, 'products': 45.35664375595009, 'manufacturing': 59.48260215636707, 'data': 158.2098558120042, 'management': 117.69403438712959, 'electronics': 190.14072167892272}), ('business', {'homepage': 296.600351160399, 'power': 167.33873480126394, 'procurement': 186.85817268086834, 'products': 35.60024729371192, 'manufacturing': 48.08031366137492, 'data': 143.26795490745917, 'management': 106.9611473622396, 'electronics': 200.88291884525808}), ('essentials', {'homepage': 1000000, 'power': 1000000, 'procurement': 1000000, 'products': 1000000, 'manufacturing': 1000000, 'data': 1000000, 'management': 1000000, 'electronics': 1000000}), ('field', {'homepage': 286.04580281951195, 'power': 163.6057157994436, 'procurement': 286.04580281951195, 'products': 44.06548682700196, 'manufacturing': 51.735649165624096, 'data': 155.87171717473382, 'management': 126.12651716118913, 'electronics': 148.62973488486054}), ('information', {'homepage': 275.8663208347937, 'power': 175.28890983760087, 'procurement': 252.86777175388158, 'products': 41.31488503378473, 'manufacturing': 71.90936863012708, 'data': 133.80374133081014, 'management': 166.7666198080487, 'electronics': 268.057649796775}), ('item', {'homepage': 321.3377689604303, 'power': 168.0053727074905, 'procurement': 253.8889410824538, 'products': 41.32107187553386, 'manufacturing': 82.31401260826237, 'data': 188.80189826006472, 'management': 188.69927656837868, 'electronics': 163.93064771806237}), ('service', {'homepage': 261.7186892039748, 'power': 163.5934530511826, 'procurement': 213.08423605499706, 'products': 42.840493892365664, 'manufacturing': 63.11668277154031, 'data': 137.7398781685014, 'management': 114.45787665121436, 'electronics': 239.94027385446734})]), (1, [('concept', {'homepage': 1000000, 'power': 133.4656687090946, 'procurement': 254.84117202298916, 'products': 46.334758549634394, 'manufacturing': 62.849185666962825, 'data': 141.9549236489962, 'management': 123.78429620998565, 'electronics': 292.4699214809868}), ('factor', {'homepage': 1000000, 'power': 158.08864515931973, 'procurement': 239.21517550458574, 'products': 42.70865782875082, 'manufacturing': 66.81365490949746, 'data': 183.3037128111998, 'management': 157.39083281819202, 'electronics': 269.5073250427175}), ('industry', {'homepage': 1000000, 'power': 126.08053055680766, 'procurement': 257.81647364541936, 'products': 35.430511292040755, 'manufacturing': 33.98694212862197, 'data': 169.38063939724324, 'management': 151.00477832707293, 'electronics': 117.57494305578957}), ('issue', {'homepage': 1000000, 'power': 146.04601405695146, 'procurement': 206.106193283641, 'products': 41.84666006748876, 'manufacturing': 69.97851560994123, 'data': 144.55103905643605, 'management': 122.32408759521132, 'electronics': 269.57570366978143}), ('production', {'homepage': 1000000, 'power': 162.8846870756111, 'procurement': 223.99241240616414, 'products': 33.3030236900373, 'manufacturing': 48.02343744887077, 'data': 171.17378490527307, 'management': 146.30253436327072, 'electronics': 186.36366294816645}), ('sector', {'homepage': 1000000, 'power': 125.50882360107347, 'procurement': 236.37047641239803, 'products': 38.6558679646223, 'manufacturing': 34.11801768409122, 'data': 201.6015701144426, 'management': 140.81318264504463, 'electronics': 133.40904078023433})]), (2, [('magazine', {'homepage': 1000000, 'power': 170.07450680273277, 'procurement': 1000000, 'products': 46.60685625631057, 'manufacturing': 1000000, 'data': 212.652084138707, 'management': 218.7089599517105, 'electronics': 177.88317784075156})])], [(0, [('cost', {'true': 1000000, 'save': 284.5037208867427, 'products': 44.88635844159002, 'overseas': 231.68509338585173, 'manufacturing': 62.41160486292912, 'china': 160.54539707716106, 'paper': 165.5605966997463, 'dfma': 1000000}), ('costs', {'true': 1000000, 'save': 1000000, 'products': 1000000, 'overseas': 1000000, 'manufacturing': 1000000, 'china': 1000000, 'paper': 1000000, 'dfma': 1000000})]), (1, [('area', {'true': 257.75063848738745, 'save': 299.56356229729846, 'products': 45.35664375595009, 'overseas': 266.88364071727733, 'manufacturing': 59.48260215636707, 'china': 114.21756463328259, 'paper': 209.1161853384097, 'dfma': 1000000}), ('business', {'true': 296.600351160399, 'save': 258.97160170240136, 'products': 35.60024729371192, 'overseas': 266.78027274042006, 'manufacturing': 48.08031366137492, 'china': 136.8003175278095, 'paper': 163.8596235999562, 'dfma': 1000000}), ('design', {'true': 188.52636252391054, 'save': 270.5109788398952, 'products': 44.66176878143506, 'overseas': 270.5109788398952, 'manufacturing': 70.29027399997912, 'china': 157.67115071200197, 'paper': 188.52636252391054, 'dfma': 1000000}), ('industry', {'true': 262.7653015233959, 'save': 287.6365520653983, 'products': 35.430511292040755, 'overseas': 276.6308483744182, 'manufacturing': 33.98694212862197, 'china': 142.7538989162788, 'paper': 129.22026607612048, 'dfma': 1000000}), ('item', {'true': 272.7033158114526, 'save': 246.08027004443503, 'products': 41.32107187553386, 'overseas': 283.70901950243274, 'manufacturing': 82.31401260826237, 'china': 129.92299348888224, 'paper': 125.70196340464688, 'dfma': 1000000}), ('product', {'true': 238.8928125788698, 'save': 234.3583958820816, 'products': 33.09845621717063, 'overseas': 290.11430855023656, 'manufacturing': 67.01973542026496, 'china': 140.51316635774455, 'paper': 138.9912059118155, 'dfma': 1000000})]), (2, [('manufacture', {'true': 233.92032070654363, 'save': 1000000, 'products': 28.847785780099695, 'overseas': 233.92032070654363, 'manufacturing': 60.367231068550865, 'china': 134.61840709690932, 'paper': 172.81259537599067, 'dfma': 1000000})])], [(0, [('area', {'machining': 255.87793702629727, 'cad': 179.84008113854821, 'fasteners': 1000000, 'products': 45.35664375595009, 'drawings': 1000000, 'gaskets': 1000000, 'cnc': 285.6980154462762, 'stamping': 310.5692659882785}), ('business', {'machining': 296.600351160399, 'cad': 189.877215154278, 'fasteners': 1000000, 'products': 35.60024729371192, 'drawings': 1000000, 'gaskets': 1000000, 'cnc': 1000000, 'stamping': 296.600351160399}), ('metal', {'machining': 242.33005150840623, 'cad': 111.8057235187532, 'fasteners': 295.1486790092973, 'products': 44.82076833823992, 'drawings': 1000000, 'gaskets': 1000000, 'cnc': 276.3343042802985, 'stamping': 242.33005150840623}), ('part', {'machining': 248.8410104076329, 'cad': 181.77181676166842, 'fasteners': 297.47546355661063, 'products': 46.1436069610199, 'drawings': 1000000, 'gaskets': 297.47546355661063, 'cnc': 253.78983828560945, 'stamping': 234.97546355661063}), ('process', {'machining': 161.9882769780195, 'cad': 205.01451585318895, 'fasteners': 1000000, 'products': 39.47786681439926, 'drawings': 1000000, 'gaskets': 1000000, 'cnc': 200.98589152650655, 'stamping': 215.48270880904914})]), (1, [('component', {'machining': 249.1603582082578, 'cad': 206.58278087228362, 'fasteners': 274.03160875026015, 'products': 43.64133745061235, 'drawings': 292.84598347925896, 'gaskets': 1000000, 'cnc': 255.21723402126133, 'stamping': 292.84598347925896})]), (2, [('database', {'machining': 1000000, 'cad': 121.83394273688393, 'fasteners': 1000000, 'products': 38.230868542738186, 'drawings': 1000000, 'gaskets': 1000000, 'cnc': 1000000, 'stamping': 1000000}), ('information', {'machining': 280.81514871277034, 'cad': 188.67663220800435, 'fasteners': 324.50077398377147, 'products': 41.31488503378473, 'drawings': 305.68639925477265, 'gaskets': 1000000, 'cnc': 1000000, 'stamping': 1000000}), ('item', {'machining': 310.3320652694503, 'cad': 201.55939717606003, 'fasteners': 275.06510086704014, 'products': 41.32107187553386, 'drawings': 283.70901950243274, 'gaskets': 340.1521436894291, 'cnc': 340.1521436894291, 'stamping': 321.3377689604303}), ('product', {'machining': 227.6143085502366, 'cad': 185.95693476555653, 'fasteners': 280.9813063203467, 'products': 33.09845621717063, 'drawings': 333.7999338212377, 'gaskets': 271.2999338212377, 'cnc': 215.54402115308275, 'stamping': 227.07679781511675}), ('services', {'machining': 1000000, 'cad': 1000000, 'fasteners': 1000000, 'products': 1000000, 'drawings': 1000000, 'gaskets': 1000000, 'cnc': 1000000, 'stamping': 1000000})])], [(0, [('approach', {'structural': 180.50815083141464, 'molecular': 165.50026894962087, 'simplest': 1000000, 'manufacturing': 70.86206292283866, 'products': 1000000, 'controlled': 230.43457971255884, 'nanotechnology': 1000000}), ('business', {'structural': 304.89174027476577, 'molecular': 1000000, 'simplest': 1000000, 'manufacturing': 54.94892989871419, 'products': 40.68599690709933, 'controlled': 304.89174027476577, 'nanotechnology': 1000000}), ('cost', {'structural': 282.1428244899944, 'molecular': 1000000, 'simplest': 1000000, 'manufacturing': 71.32754841477612, 'products': 51.29869536181716, 'controlled': 282.1428244899944, 'nanotechnology': 1000000}), ('extra', {'structural': 1000000, 'molecular': 1000000, 'simplest': 1000000, 'manufacturing': 94.70008287793891, 'products': 1000000, 'controlled': 1000000, 'nanotechnology': 1000000}), ('factor', {'structural': 223.56949706284007, 'molecular': 240.3972514221747, 'simplest': 1000000, 'manufacturing': 76.35846275371136, 'products': 48.80989466142951, 'controlled': 268.4544206378623, 'nanotechnology': 1000000})]), (1, [('area', {'structural': 238.6228089830103, 'molecular': 251.56787317644543, 'simplest': 1000000, 'manufacturing': 67.98011675013377, 'products': 51.83616429251438, 'controlled': 292.43192803005394, 'nanotechnology': 241.2888320561397}), ('design', {'structural': 212.57093880906854, 'molecular': 309.1554043884516, 'simplest': 1000000, 'manufacturing': 80.33174171426185, 'products': 51.0420214644972, 'controlled': 275.07531476561854, 'nanotechnology': 309.1554043884516}), ('industry', {'structural': 275.844438825152, 'molecular': 307.22534552731366, 'simplest': 1000000, 'manufacturing': 38.84221957556797, 'products': 40.49201290518943, 'controlled': 328.72748807474085, 'nanotechnology': 249.16010576710966}), ('product', {'structural': 256.61506529525735, 'molecular': 288.55492467684445, 'simplest': 1000000, 'manufacturing': 76.59398333744566, 'products': 37.826807105337856, 'controlled': 278.1172078426846, 'nanotechnology': 338.4813535579887}), ('software', {'structural': 267.40335213931894, 'molecular': 250.04349151886936, 'simplest': 1000000, 'manufacturing': 83.34783050628978, 'products': 50.24137229297098, 'controlled': 276.3275476113404, 'nanotechnology': 310.4076372341734}), ('technology', {'structural': 244.31879120870786, 'molecular': 205.62135624865704, 'simplest': 1000000, 'manufacturing': 70.07956719529163, 'products': 48.995397338094456, 'controlled': 233.71737460457766, 'nanotechnology': 183.62810527379784})]), (2, [('field', {'structural': 222.45248908665963, 'molecular': 188.7310281334731, 'simplest': 1000000, 'manufacturing': 59.12645618928467, 'products': 50.36055637371653, 'controlled': 305.40734638915785, 'nanotechnology': 195.5631253124954})])], [(0, [('business', {'epa': 127.43439905789411, 'manufactured': 1000000, 'wrr': 1000000, 'buy': 157.91753508746896, 'recycled': 1000000, 'products': 31.644664261077253, 'organizations': 224.81308967946475, 'gov': 99.44102882269529, 'percentage': 246.9208679390224})]), (1, [('factor', {'epa': 145.7661704162118, 'manufactured': 277.1561561282415, 'wrr': 1000000, 'buy': 187.70462206846398, 'recycled': 245.2655301433682, 'products': 37.963251403334056, 'organizations': 250.6494197549269, 'gov': 94.45438746378652, 'percentage': 177.77420525182944}), ('term', {'epa': 138.3177232130105, 'manufactured': 263.8772447677369, 'wrr': 1000000, 'buy': 168.08414090102636, 'recycled': 247.1533561197379, 'products': 38.855559786134535, 'organizations': 263.8772447677369, 'gov': 96.66338220066766, 'percentage': 198.5388414868657})]), (2, [('content', {'epa': 157.79363165484617, 'manufactured': 1000000, 'wrr': 1000000, 'buy': 182.63574450719167, 'recycled': 221.46741141474826, 'products': 40.26680207540876, 'organizations': 1000000, 'gov': 99.45864480746248, 'percentage': 1000000}), ('item', {'epa': 145.89961105772687, 'manufactured': 233.99918764736628, 'wrr': 1000000, 'buy': 212.17694492405369, 'recycled': 205.23590400137036, 'products': 36.72984166714121, 'organizations': 1000000, 'gov': 131.28915584223796, 'percentage': 222.83725193732664}), ('manufacture', {'epa': 214.87021488405546, 'manufactured': 179.49120946254578, 'wrr': 1000000, 'buy': 241.37695125737002, 'recycled': 207.92917396137213, 'products': 25.642476248977502, 'organizations': 1000000, 'gov': 149.76870840624736, 'percentage': 1000000})])], [(0, [('business', {'cast': 126.66201028387687, 'pre': 91.40026313534261, 'health': 107.11088791826602, 'portland': 183.93798246501663, 'osha': 183.93798246501663, 'cement': 136.24625416182656, 'products': 25.891088940881396, 'topics': 1000000, 'phases': 1000000, 'gov': 81.36084176402342, 'safety': 156.57161922283655}), ('manufacture', {'cast': 141.56073642065394, 'pre': 103.31377796938798, 'health': 141.56073642065394, 'portland': 183.80705122584905, 'osha': 175.80290308695447, 'cement': 136.4710200822934, 'products': 20.980207840072506, 'topics': 1000000, 'phases': 1000000, 'gov': 122.53803415056603, 'safety': 156.44068798366897}), ('material', {'cast': 114.42425544701574, 'pre': 97.1956423883713, 'health': 165.8460242942575, 'portland': 165.8460242942575, 'osha': 1000000, 'cement': 108.34168719570414, 'products': 31.8832409651405, 'topics': 208.0923390994527, 'phases': 1000000, 'gov': 108.10023014533894, 'safety': 158.63160779488211}), ('production', {'cast': 108.17084617466823, 'pre': 83.26500684905398, 'health': 141.2162428990438, 'portland': 176.58675428011856, 'osha': 1000000, 'cement': 129.25072313656284, 'products': 24.220380865481673, 'topics': 1000000, 'phases': 1000000, 'gov': 103.26628301342258, 'safety': 119.52891313905913}), ('sector', {'cast': 126.45125557265315, 'pre': 118.18563793879986, 'health': 62.50213420678384, 'portland': 203.677164860654, 'osha': 1000000, 'cement': 108.20577644621933, 'products': 28.113358519725313, 'topics': 1000000, 'phases': 1000000, 'gov': 63.49367737382047, 'safety': 139.3601944983441})]), (1, [('index', {'cast': 1000000, 'pre': 84.24299640743979, 'health': 101.31861522384366, 'portland': 1000000, 'osha': 1000000, 'cement': 1000000, 'products': 1000000, 'topics': 1000000, 'phases': 1000000, 'gov': 94.91210672474256, 'safety': 134.36401194821926})]), (2, [('area', {'cast': 125.01410262275994, 'pre': 93.92226307864556, 'health': 87.08353132413988, 'portland': 166.24338157985008, 'osha': 182.49389756241504, 'cement': 132.89933077389995, 'products': 32.98665000432734, 'topics': 212.18537546129434, 'phases': 225.8685570823844, 'gov': 79.0115278725716, 'safety': 114.59888137364891}), ('factor', {'cast': 150.13950325936742, 'pre': 72.96249736120305, 'health': 96.19170673973846, 'portland': 254.1304909834686, 'osha': 226.7641277412885, 'cement': 123.28859118795921, 'products': 31.060842057273323, 'topics': 218.75997960239388, 'phases': 232.44316122348394, 'gov': 77.28086247037079, 'safety': 120.53868296307466}), ('information', {'cast': 124.83673547576117, 'pre': 85.12263998408325, 'health': 115.44391103350094, 'portland': 236.00056289728838, 'osha': 222.31738127619832, 'cement': 125.92765713047784, 'products': 30.047189115479807, 'topics': 186.9468698951236, 'phases': 236.00056289728838, 'gov': 91.5682020710723, 'safety': 132.6297515671211}), ('issue', {'cast': 144.08693241168774, 'pre': 76.46915086753674, 'health': 81.77390323189881, 'portland': 221.70375859795755, 'osha': 196.05505721438652, 'cement': 117.9860023398525, 'products': 30.43393459453728, 'topics': 211.61972452448686, 'phases': 216.0247251157621, 'gov': 69.63879453926147, 'safety': 97.15378582864591}), ('issues', {'cast': 1000000, 'pre': 1000000, 'health': 78.00015198339997, 'portland': 1000000, 'osha': 1000000, 'cement': 1000000, 'products': 1000000, 'topics': 1000000, 'phases': 1000000, 'gov': 1000000, 'safety': 1000000}), ('topic', {'cast': 136.91218070462236, 'pre': 77.29919832061552, 'health': 86.72551609014238, 'portland': 209.48477596037054, 'osha': 186.2173504613308, 'cement': 127.85386608109441, 'products': 32.13641130348922, 'topics': 184.75441030808528, 'phases': 186.2173504613308, 'gov': 76.87221405521292, 'safety': 101.6879745013117})])], [(0, [('cleaners', {'heaters': 1000000, 'pumps': 1000000, 'filters': 1000000, 'pentair': 1000000, 'products': 1000000, 'welcome': 1000000, 'california': 1000000, 'pool': 1000000}), ('item', {'heaters': 340.1521436894291, 'pumps': 283.70901950243274, 'filters': 264.8946447734339, 'pentair': 340.1521436894291, 'products': 41.32107187553386, 'welcome': 272.7033158114526, 'california': 261.69761212047257, 'pool': 185.24173771481395}), ('sanitizers', {'heaters': 1000000, 'pumps': 1000000, 'filters': 1000000, 'pentair': 1000000, 'products': 1000000, 'welcome': 1000000, 'california': 1000000, 'pool': 118.94312418699646})]), (1, [('area', {'heaters': 1000000, 'pumps': 1000000, 'filters': 1000000, 'pentair': 1000000, 'products': 45.35664375595009, 'welcome': 1000000, 'california': 155.4343475250737, 'pool': 173.15319805362708}), ('business', {'heaters': 266.78027274042006, 'pumps': 296.600351160399, 'filters': 1000000, 'pentair': 1000000, 'products': 35.60024729371192, 'welcome': 266.78027274042006, 'california': 208.1645169122229, 'pool': 206.15297420151032}), ('equipment', {'heaters': 260.44440491229113, 'pumps': 200.80424807233328, 'filters': 1000000, 'pentair': 1000000, 'products': 39.41068720750725, 'welcome': 1000000, 'california': 241.63003018329226, 'pool': 181.98987334333447}), ('manufacturer', {'heaters': 1000000, 'pumps': 216.01474096465853, 'filters': 268.8333684655496, 'pentair': 268.8333684655496, 'products': 28.43174256812741, 'welcome': 1000000, 'california': 216.01474096465853, 'pool': 177.43349359686488})]), (2, [('component', {'heaters': 274.03160875026015, 'pumps': 221.2129812493691, 'filters': 236.4028592922625, 'pentair': 1000000, 'products': 43.64133745061235, 'welcome': 1000000, 'california': 1000000, 'pool': 198.7741098342649})])], [(0, [('area', {'chennai': 233.40391300742374, 'infoworld': 1000000, 'news': 170.64028775225844, 'manufacturing': 47.586081725093656, 'products': 36.28531500476007, 'india': 134.0653262941115, 'cisco': 149.00518200569596, 'south': 97.66647497117074, 'systems': 151.98446650490814, 'domestic': 161.24076363448899}), ('business', {'chennai': 222.22878114512014, 'infoworld': 1000000, 'news': 131.9197824459258, 'manufacturing': 38.46425092909994, 'products': 28.480197834969534, 'india': 152.77047692689354, 'cisco': 150.26214645360702, 'south': 149.48653814469466, 'systems': 133.68618056301295, 'domestic': 177.07428179552298}), ('facility', {'chennai': 208.3845697806924, 'infoworld': 1000000, 'news': 153.251452685835, 'manufacturing': 50.53381322689625, 'products': 1000000, 'india': 176.54346490133366, 'cisco': 193.33306999749334, 'south': 149.58000682790833, 'systems': 182.2406325166755, 'domestic': 187.08613316707832}), ('factor', {'chennai': 279.5435400818155, 'infoworld': 1000000, 'news': 194.16503127691865, 'manufacturing': 53.45092392759797, 'products': 34.166926263000654, 'india': 192.52540560710324, 'cisco': 279.5435400818155, 'south': 192.13413873150543, 'systems': 192.13413873150543, 'domestic': 152.2164069079089}), ('market', {'chennai': 197.2507671561976, 'infoworld': 1000000, 'news': 112.47668333168774, 'manufacturing': 50.00887784956306, 'products': 30.806480385060592, 'india': 74.78456250126126, 'cisco': 125.68257894824823, 'south': 95.82220452157073, 'systems': 159.8413658058876, 'domestic': 156.94176845700326})]), (1, [('article', {'chennai': 1000000, 'infoworld': 1000000, 'news': 109.37356213420497, 'manufacturing': 57.77041406061548, 'products': 30.375328245714805, 'india': 154.9124029171167, 'cisco': 197.16730491782954, 'south': 173.3112421818464, 'systems': 182.11580513463048, 'domestic': 197.16730491782954}), ('information', {'chennai': 1000000, 'infoworld': 1000000, 'news': 121.27997683139719, 'manufacturing': 57.52749490410166, 'products': 33.051908027027785, 'india': 194.5491194038181, 'cisco': 198.07817311810348, 'south': 200.7960562342331, 'systems': 166.13903320046836, 'domestic': 189.70361875341527}), ('manufacture', {'chennai': 1000000, 'infoworld': 1000000, 'news': 202.18775634843396, 'manufacturing': 48.293784854840695, 'products': 23.07822862407976, 'india': 148.2286940460527, 'cisco': 118.35307586719064, 'south': 143.3831933956499, 'systems': 119.52713065966678, 'domestic': 148.2286940460527}), ('topic', {'chennai': 1000000, 'infoworld': 1000000, 'news': 163.06675279056006, 'manufacturing': 54.94852016787329, 'products': 35.35005243383814, 'india': 163.06675279056006, 'cisco': 218.28135112209287, 'south': 176.47419125402635, 'systems': 145.2686846504745, 'domestic': 148.384350688491})]), (2, [('service', {'chennai': 1000000, 'infoworld': 1000000, 'news': 136.25505146823204, 'manufacturing': 50.49334621723226, 'products': 34.272395113892536, 'india': 164.01265402502477, 'cisco': 186.66203277517192, 'south': 195.1751185449198, 'systems': 155.13926463080054, 'domestic': 150.12946515803637})])], [(0, [('item', {'clients': 413.7760870259337, 'products': 55.0947625007118, 'manufacturing': 109.75201681101652, 'precision': 358.0255219460524, 'multi': 266.0011913934977, 'spindle': 330.4427536926003}), ('machine', {'clients': 338.95666866095684, 'products': 54.399375291088184, 'manufacturing': 86.47881984700439, 'precision': 212.41384033779957, 'multi': 204.55801393431224, 'spindle': 206.36795140874858}), ('product', {'clients': 394.8949124843201, 'products': 44.13127495622749, 'manufacturing': 89.35964722701996, 'precision': 273.65784081694426, 'multi': 227.33459036833904, 'spindle': 301.37692262825124}), ('services', {'clients': 1000000, 'products': 1000000, 'manufacturing': 1000000, 'precision': 1000000, 'multi': 1000000, 'spindle': 1000000})]), (1, [('area', {'clients': 439.17818762303637, 'products': 60.47552500793344, 'manufacturing': 79.31013620848942, 'precision': 322.6831869003666, 'multi': 240.22903542814225, 'spindle': 1000000}), ('company', {'clients': 374.4432173526751, 'products': 41.45720816019658, 'manufacturing': 80.3127393584483, 'precision': 301.5214454033667, 'multi': 239.89774211300664, 'spindle': 439.2891548846454}), ('factor', {'clients': 361.29985804441685, 'products': 56.94487710500109, 'manufacturing': 89.08487321266327, 'precision': 324.3250664416908, 'multi': 237.41097699225918, 'spindle': 355.72095890853245}), ('information', {'clients': 357.41019972903325, 'products': 55.08651337837963, 'manufacturing': 95.87915817350276, 'precision': 332.32436675703485, 'multi': 258.2158915481757, 'spindle': 382.49603270103177})]), (2, [('business', {'clients': 1000000, 'products': 47.466996391615886, 'manufacturing': 64.10708488183323, 'precision': 308.68441111734654, 'multi': 234.25893061023473, 'spindle': 395.467134880532}), ('production', {'clients': 1000000, 'products': 44.40403158671639, 'manufacturing': 64.0312499318277, 'precision': 265.494882485549, 'multi': 233.81061234291516, 'spindle': 298.65654987488546})])]]
		wl = ['china', 'manufacturers', 'products', 'suppliers', 'data', 'electronics', 'homepage', 'management', 'manufacturing', 'power', 'procurement', 'products', 'china', 'dfma', 'manufacturing', 'overseas', 'paper', 'products', 'save', 'true', 'cad', 'cnc', 'drawings', 'fasteners', 'gaskets', 'machining', 'products', 'stamping', 'controlled', 'manufacturing', 'molecular', 'nanotechnology', 'products', 'simplest', 'structural', 'buy', 'epa', 'gov', 'manufactured', 'organizations', 'percentage', 'products', 'recycled', 'wrr', 'cast', 'cement', 'gov', 'health', 'osha', 'phases', 'portland', 'pre', 'products', 'safety', 'topics', 'california', 'filters', 'heaters', 'pentair', 'pool', 'products', 'pumps', 'welcome', 'chennai', 'cisco', 'domestic', 'india', 'infoworld', 'manufacturing', 'news', 'products', 'south', 'systems', 'clients', 'manufacturing', 'multi', 'precision', 'products', 'spindle']
	#print("CS : \n\n\n\n\n",cs)
	#print("WL : \n\n\n\n\n",wl)
	else:
		cs,wl = getSense(F)
	wl = sorted(wl)
	wlen = len(wl)
	#expansion of vectors
	expvecl = []
	for i in cs:
		V,el = getConceptVecEntityList(i)
		expvec = getExpandedVector(V,el,wl)
		expvecl.append([v for k,v in expvec])
	km = KMeans(expvecl,wlen)
	#print("indices\n\n",km)
	clusterofChunk = {}
	for i,x in enumerate(km):
		cluster = []
		for y in x:
			cluster.append(cs[y])
		clusterofChunk.update({i:cluster})
	#print("\n\n\nclusterofChunk\n\n",clusterofChunk)
	return clusterofChunk


def shortTextToChunkClusterMatch(F,st):
	coc = getChunkCluster()
	print(coc.items())
	tcost = 1000000000000
	epsilon = (float(0.9999)/tcost)
	ind = -1
	for j,docClu in coc.items():
		cost=0
		if j==0:
			print("DocClu Schema :",docClu,"\n\n\n\n\n")
		for i,doc in enumerate(docClu):
			cost+= SemDistShortText(doc,st)
		cost = cost / (len(docClu)+epsilon)
		if cost < tcost:
			tcost = cost
			ind = j
	return tcost






if __name__ == "__main__":
	st =  [
	(
		0, 
		[
			('business', {'multi': 234.25893061023473, 'products': 47.466996391615886, 'precision': 308.68441111734654, 'clients': 1000000, 'manufacturing': 64.10708488183323, 'spindle': 395.467134880532}), 
			('production', {'multi': 233.81061234291516, 'products': 44.40403158671639, 'precision': 265.494882485549, 'clients': 1000000, 'manufacturing': 64.0312499318277, 'spindle': 298.65654987488546})
		]
	), 
	(
		1, 
		[
			('area', {'multi': 240.22903542814225, 'products': 60.47552500793344, 'precision': 322.6831869003666, 'clients': 439.17818762303637, 'manufacturing': 79.31013620848942, 'spindle': 1000000}), 
			('company', {'multi': 239.89774211300664, 'products': 41.45720816019658, 'precision': 301.5214454033667, 'clients': 374.4432173526751, 'manufacturing': 80.3127393584483, 'spindle': 439.2891548846454}), 
			('factor', {'multi': 237.41097699225918, 'products': 56.94487710500109, 'precision': 324.3250664416908, 'clients': 361.29985804441685, 'manufacturing': 89.08487321266327, 'spindle': 355.72095890853245}), 
			('information', {'multi': 258.2158915481757, 'products': 55.08651337837963, 'precision': 332.32436675703485, 'clients': 357.41019972903325, 'manufacturing': 95.87915817350276, 'spindle': 382.49603270103177})
		]
	), 
	(
		2, 
		[
			('item', {'multi': 266.0011913934977, 'products': 55.0947625007118, 'precision': 358.0255219460524, 'clients': 413.7760870259337, 'manufacturing': 109.75201681101652, 'spindle': 330.4427536926003}), 
			('machine', {'multi': 204.55801393431224, 'products': 54.399375291088184, 'precision': 212.41384033779957, 'clients': 338.95666866095684, 'manufacturing': 86.47881984700439, 'spindle': 206.36795140874858}), 
			('product', {'multi': 227.33459036833904, 'products': 44.13127495622749, 'precision': 273.65784081694426, 'clients': 394.8949124843201, 'manufacturing': 89.35964722701996, 'spindle': 301.37692262825124}), 
			('services', {'multi': 1000000, 'products': 1000000, 'precision': 1000000, 'clients': 1000000, 'manufacturing': 1000000, 'spindle': 1000000})
		]
	)
]
	#test: getConceptVecEntityList
	#getConceptVecEntityList(st)
	#test: SemDistShortText
	#print(SemDistShortText(st,st))
	start = time.time()
	cost = shortTextToChunkClusterMatch(None,st)
	print(cost)
	end -= time.time()
	print(end-start)