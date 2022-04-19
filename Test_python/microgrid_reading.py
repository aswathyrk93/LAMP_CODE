from pymodbus.client.sync import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder
import time
import Fault_Classifier_testC1 as svC1
import Fault_Classifier_testC2 as svC2
import Fault_Classifier_testC3 as svC3
import Fault_Classifier_testC4 as svC4
import GP_OCSVM_TEST_C1 as GPC1
import GP_OCSVM_TEST_C2 as GPC2
import GP_OCSVM_TEST_C3 as GPC3
import GP_OCSVM_TEST_C4 as GPC4
import CNN_TEST as cn
import numpy as np
import sys

def computing_fin_prob(l1, l11, relay):
        
        pfc_C1 = svC1.svm_test_main(l1, relay)
        pfc_C2 = svC2.svm_test_main(l1, relay)
        pfc_C3 = svC3.svm_test_main(l1, relay)
        pfc_C4 = svC4.svm_test_main(l1, relay)
        pGP_C1 = GPC1.GP_test_main(l1, relay)
        pGP_C2 = GPC2.GP_test_main(l1, relay)
        pGP_C3 = GPC3.GP_test_main(l1, relay)
        pGP_C4 = GPC4.GP_test_main(l1, relay)
        pCNN = cn.CNN_TEST1(np.array(l11), relay)
        
        # computing the marginal probabilities for fault detector and fault classifier
        PFC1 = pfc_C1[0]*pCNN[0] + pfc_C2[0]*pCNN[1] + pfc_C3[0]*pCNN[2] + pfc_C4[0]*pCNN[3]
        PFC2 = pfc_C1[1]*pCNN[0] + pfc_C2[1]*pCNN[1] + pfc_C3[1]*pCNN[2] + pfc_C4[1]*pCNN[3]
        PFC3 = pfc_C1[2]*pCNN[0] + pfc_C2[2]*pCNN[1] + pfc_C3[2]*pCNN[2] + pfc_C4[2]*pCNN[3]
        PFC = [PFC1, PFC2, PFC3]

        PF = pGP_C1*pCNN[0] + pGP_C2*pCNN[1] + pGP_C3*pCNN[2] + pGP_C4*pCNN[3]

        return pCNN, PF, PFC


#[V1, V2, V0, I1, I2, I0, fault status_boolean (0 or 1), fault type (2 fro AB as Adams doc), fault_resistance (0.5), RTL1, RTL2, RTL3, RTL4]

# modbus
client = ModbusTcpClient('192.168.1.19', port=502)
print(client.connect())
print(client.is_socket_open())
l11 = []
i=0
while(client.connect()):  
        time.sleep(1)
        try:

                result = client.read_holding_registers(0,26,unit =1) 
                decoder = BinaryPayloadDecoder.fromRegisters(result.registers, byteorder=Endian.Little, wordorder=Endian.Little)
	
                array = [decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float(),decoder.decode_32bit_float()]
                
                l1 = [array[0], array[1], array[2],array[3], array[4], array[5]]     #voltage and current values saved here
                relay = 'RTL3'
                l11.append(l1)
                print("\n")
                i = i+1
                print('iteration completed = ', i)
                print(array)
                [pCNN, PF, PFC] = computing_fin_prob(l1, l11, relay)
                print('Configuration classifier (1DCNN): ',pCNN)
                print('Fault Detector (GP+OCSVM): ',PF)
                print('Fault classifier (SVM): ',PFC)
	        # result1 = client.read_holding_registers()
        except KeyboardInterrupt:
                print('Keyboard Interrupt')
                client.close()


