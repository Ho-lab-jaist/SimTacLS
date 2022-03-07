import Sofa
import math
import csv
import os
import numpy as np
from array import array


# lc = 12
outer_skin_indices = [7162L, 7163L, 7164L, 7165L, 7166L, 7167L, 7168L, 7169L, 7170L, 7171L, 7172L, 7173L, 7174L, 7175L, 7176L, 7177L, 7178L, 7179L, 7180L, 7181L, 7182L, 7183L, 7184L, 7185L, 7186L, 7187L, 7188L, 7189L, 7190L, 7191L, 7192L, 7193L, 7194L, 7195L, 7196L, 7197L, 7198L, 7199L, 7200L, 7201L, 7202L, 7203L, 7204L, 7205L, 7206L, 7207L, 7208L, 7209L, 7210L, 7211L, 7212L, 7213L, 7214L, 7215L, 7216L, 7217L, 7218L, 7219L, 7220L, 7221L, 7222L, 7223L, 7224L, 7225L, 7226L, 7227L, 7228L, 7229L, 7230L, 7231L, 7232L, 7233L, 7234L, 7235L, 7236L, 7237L, 7238L, 7239L, 7240L, 7241L, 7242L, 7243L, 7244L, 7245L, 7246L, 7247L, 7248L, 7249L, 7250L, 7251L, 7252L, 7253L, 7254L, 7255L, 7256L, 7257L, 7258L, 7259L, 7260L, 7261L, 7262L, 7263L, 7264L, 7265L, 7266L, 7267L, 7268L, 7269L, 7270L, 7271L, 7272L, 7273L, 7274L, 7275L, 7276L, 7277L, 7278L, 7279L, 7280L, 7281L, 7282L, 7283L, 7284L, 7285L, 7286L, 7287L, 7288L, 7289L, 7290L, 7291L, 7292L, 7293L, 7294L, 7295L, 7296L, 7297L, 7298L, 7299L, 7300L, 7301L, 7302L, 7303L, 7304L, 7305L, 7306L, 7307L, 7308L, 7309L, 7310L, 7311L, 7312L, 7313L, 7314L, 7315L, 7316L, 7317L, 7318L, 7319L, 7320L, 7321L, 7322L, 7323L, 7324L, 7325L, 7326L, 7327L, 7328L, 7329L, 7330L, 7331L, 7332L, 7333L, 7334L, 7335L, 7336L, 7337L, 7338L, 7339L, 7340L, 7341L, 7342L, 7343L, 7344L, 7345L, 7346L, 7347L, 7348L, 7349L, 7350L, 7351L, 7352L, 7353L, 7354L, 7355L, 7356L, 7357L, 7358L, 7359L, 7360L, 7361L, 7362L, 7363L, 7364L, 7365L, 7366L, 7367L, 7368L, 7369L, 7370L, 7371L, 7372L, 7373L, 7374L, 7375L, 7376L, 7377L, 7378L, 7379L, 7380L, 7381L, 7382L, 7383L, 7384L, 7385L, 7386L, 7387L, 7388L, 7389L, 7390L, 7391L, 7392L, 7393L, 7394L, 7395L, 7396L, 7397L, 7398L, 7399L, 7400L, 7401L, 7402L, 7403L, 7404L, 7405L, 7406L, 7407L, 7408L, 7409L, 7410L, 7411L, 7412L, 7413L, 7414L, 7415L, 7416L, 7417L, 7418L, 7419L, 7420L, 7421L, 7422L, 7423L, 7424L, 7425L, 7426L, 7427L, 7428L, 7429L, 7430L, 7431L, 7432L, 7433L, 7434L, 7435L, 7436L, 7437L, 7438L, 7439L, 7440L, 7441L, 7442L, 7443L, 7444L, 7445L, 7446L, 7447L, 7448L, 7449L, 7450L, 7451L, 7452L, 7453L, 7454L, 7455L, 7456L, 7457L, 7458L, 7459L, 7460L, 7461L, 7462L, 7463L, 7464L, 7465L, 7466L, 7467L, 7468L, 7469L, 7470L, 7471L, 7472L, 7473L, 2623L, 2624L, 2625L, 2626L, 2627L, 2628L, 2629L, 2630L, 2631L, 2632L, 2633L, 2634L, 2635L, 2636L, 2637L, 2638L, 2639L, 2640L, 2641L, 2642L, 2643L, 2614L, 2615L, 2616L, 2617L, 2618L, 2619L, 2620L, 2621L, 2622L, 2644L, 2645L, 2646L, 2647L, 2648L, 2649L, 2650L, 2651L, 2652L, 2653L, 2654L, 2655L, 2656L, 2657L, 2658L, 2659L, 2660L, 2661L, 2662L, 2663L, 2664L, 2665L, 2666L, 2667L, 2668L, 2669L, 2670L, 2671L, 2672L, 2673L, 1029L, 1030L, 1031L, 1032L, 7474L, 7475L, 7476L, 7477L, 7478L, 7479L, 7480L, 7481L, 7482L, 7483L, 7484L, 7485L, 7486L, 7487L, 7488L, 7489L, 7490L, 7491L, 7492L, 7493L, 7494L, 7495L, 7496L, 7497L, 7498L, 7499L, 7500L, 7501L, 7502L, 7503L, 7504L, 7505L, 7506L, 7507L, 7508L, 7509L, 7510L, 7511L, 7512L, 7513L, 7514L, 7515L, 7516L, 7517L, 7518L, 7519L, 7520L, 7521L, 7522L, 7523L, 7524L, 7525L, 7526L, 7527L, 7528L, 7529L, 7530L, 7531L, 7532L, 7533L, 7534L, 7535L, 7536L, 7537L, 7538L, 7539L, 7540L, 7541L, 7542L, 7543L, 7544L, 7545L, 7546L, 7547L, 7548L, 7549L, 7550L, 7551L, 7552L, 7553L, 7554L, 7555L, 7556L, 7557L, 7558L, 7559L, 7560L, 7561L, 7562L, 7563L, 7564L, 7565L, 7566L, 7567L, 7568L, 7569L, 7570L, 7571L, 7572L, 7573L, 7574L, 7575L, 7576L, 7577L, 7578L, 7579L, 7580L, 7581L, 7582L, 7583L, 7584L, 7585L, 7586L, 7587L, 7588L, 7589L, 7590L, 7591L, 7592L, 7593L, 7594L, 7595L, 7596L, 7597L, 7598L, 7599L, 7600L, 7601L, 7602L, 7603L, 7604L, 7605L, 7606L, 7607L, 7608L, 7609L, 7610L, 7611L, 7612L, 7613L, 7614L, 7615L, 7616L, 7617L, 7618L, 7619L, 7620L, 7621L, 7622L, 7623L, 7624L, 7625L, 7626L, 7627L, 7628L, 7629L, 7630L, 7631L, 7632L, 7633L, 7634L, 7635L, 7636L, 7637L, 7638L, 7639L, 7640L, 7641L, 7642L, 7643L, 7644L, 7645L, 7646L, 7647L, 7648L, 7649L, 7650L, 7651L, 7652L, 7653L, 7654L, 7655L, 7656L, 7657L, 7658L, 7659L, 7660L, 7661L, 7662L, 7663L, 7664L, 7665L, 7666L, 7667L, 7668L, 7669L, 7670L, 7671L, 7672L, 7673L, 7674L, 7675L, 7676L, 7677L, 7678L, 7679L, 7680L, 7681L, 7682L, 7683L, 7684L, 7685L, 7686L, 7687L, 7688L, 7689L, 7690L, 7691L, 7692L, 7693L, 7694L, 7695L, 7696L, 7697L, 7698L, 7699L, 7700L, 7701L, 7702L, 7703L, 7704L, 7705L, 7706L, 7707L, 7708L, 7709L, 7710L, 7711L, 7712L, 7713L, 7714L, 7715L, 7716L, 7717L, 7718L, 7719L, 7720L, 7721L, 7722L, 7723L, 7724L, 7725L, 7726L, 7727L, 7728L, 7729L, 7730L, 7731L, 7732L, 7733L, 7734L, 7735L, 7736L, 7737L, 7738L, 7739L, 7740L, 7741L, 7742L, 7743L, 7744L, 7745L, 7746L, 7747L, 7748L, 7749L, 7750L, 7751L, 7752L, 7753L, 7754L, 7755L, 7756L, 7757L, 7758L, 7759L, 7760L, 7761L, 7762L, 7763L, 7764L, 7765L, 7766L, 7767L, 7768L, 7769L, 7770L, 7771L, 7772L, 7773L, 7774L, 7775L, 7776L, 7777L, 7778L, 7779L, 7780L, 7781L, 7782L, 7783L, 7784L, 7785L, 7786L, 2605L, 2606L, 2607L, 2608L, 2609L, 2610L, 2611L, 2612L, 2613L, 2674L, 2675L, 2676L, 2677L, 2678L, 2679L, 2680L, 2681L, 2682L]
for i in range(len(outer_skin_indices)):
    outer_skin_indices[i] = int(outer_skin_indices[i])-1

class controller(Sofa.PythonScriptController):

    # For Rigin3d
    def moveRestPos(self, rest_pos, dx, dy, dz):
        str_out = ' '
        for i in xrange(0, len(rest_pos)):
            str_out = str_out + ' ' + str(rest_pos[i][0] + dx)
            str_out = str_out + ' ' + str(rest_pos[i][1] + dy)
            str_out = str_out + ' ' + str(rest_pos[i][2] + dz)
            str_out = str_out + ' ' + str(rest_pos[i][3])
            str_out = str_out + ' ' + str(rest_pos[i][4])
            str_out = str_out + ' ' + str(rest_pos[i][5])
            str_out = str_out + ' ' + str(rest_pos[i][6])
        return str_out

    def changeRestPos(self, rest_pos, x, y, z):
        str_out = ' '
        for i in xrange(0, len(rest_pos)):
            str_out = str_out + ' ' + str(x)
            str_out = str_out + ' ' + str(y)
            str_out = str_out + ' ' + str(z)
            str_out = str_out + ' ' + str(rest_pos[i][3])
            str_out = str_out + ' ' + str(rest_pos[i][4])
            str_out = str_out + ' ' + str(rest_pos[i][5])
            str_out = str_out + ' ' + str(rest_pos[i][6])

        return str_out

    def rotateRestPos(self, rest_pos, rx, centerPosX, centerPosY):
        str_out = ' '
        for i in xrange(0, len(rest_pos)):
            newRestPosX = (rest_pos[i][0] - centerPosX) * math.cos(rx) - (rest_pos[i][1] - centerPosY) * math.sin(
                rx) + centerPosX
            newRestPosY = (rest_pos[i][0] - centerPosX) * math.sin(rx) + (rest_pos[i][1] - centerPosY) * math.cos(
                rx) + centerPosY
            str_out = str_out + ' ' + str(newRestPosX)
            str_out = str_out + ' ' + str(newRestPosY)
            str_out = str_out + ' ' + str(rest_pos[i][2])
        return str_out




    def indention(self, x, y):
        new_pos = self.moveRestPos(self.MecaObjectfinger.rest_position, x, y, 0.0)
        self.MecaObjectfinger.findData('rest_position').value = new_pos
        return new_pos

    def movevertical(self, z):
        new_pos = self.moveRestPos(self.MecaObjectfinger.rest_position, 0.0, 0.0, z)
        self.MecaObjectfinger.findData('rest_position').value = new_pos
        return new_pos

    def movenext(self, x, y, z):
        str_out = ' '
        eulerz = math.atan2(y,x)
        cz = math.cos(eulerz*0.5)
        sz = math.sin(eulerz*0.5)
        cy = math.cos(90 * 0.5 * math.pi / 180.0)
        sy = math.sin(90 * 0.5 * math.pi / 180.0)
        cx = math.cos(0 * 0.5 * math.pi / 180.0)
        sx = math.sin(0 * 0.5 * math.pi / 180.0)
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
        w = cx * cy * cz + sx * sy * sz
        for i in xrange(0, len(self.original_finger_pos)):
            str_out = str_out + ' ' + str(x+(self.dist_contact+self.pre_gap)*math.cos(eulerz))
            str_out = str_out + ' ' + str(y+(self.dist_contact+self.pre_gap)*math.sin(eulerz))
            str_out = str_out + ' ' + str(z)
            str_out = str_out + ' ' + str(qx)
            str_out = str_out + ' ' + str(qy)
            str_out = str_out + ' ' + str(qz)
            str_out = str_out + ' ' + str(w)
        self.MecaObjectfinger.findData('position').value = str_out
        self.MecaObjectfinger.findData('rest_position').value = str_out
        self.rotAngle = eulerz
        return eulerz



    def rotatefinger(self, rest_pos, rx, eulerx, eulery, eulerz, centerPosX, centerPosY):  #roll(X) pitch(Y) yaw(Z)
        str_out = ' '
        cz = math.cos(eulerz*0.5*math.pi/180.0)
        sz = math.sin(eulerz*0.5*math.pi/180.0)
        cy = math.cos(eulery*0.5*math.pi/180.0)
        sy = math.sin(eulery * 0.5*math.pi/180.0)
        cx = math.cos(eulerx * 0.5*math.pi/180.0)
        sx = math.sin(eulerx * 0.5*math.pi/180.0)
        qx = sx*cy*cz-cx*sy*sz
        qy = cx*sy*cz+sx*cy*sz
        qz = cx*cy*sz-sx*sy*cz
        w = cx*cy*cz+sx*sy*sz

        for i in xrange(0, len(rest_pos)):
            newRestPosX = (rest_pos[i][0] - centerPosX) * math.cos(rx) - (rest_pos[i][1] - centerPosY) * math.sin(
                rx) + centerPosX
            newRestPosY = (rest_pos[i][0] - centerPosX) * math.sin(rx) + (rest_pos[i][1] - centerPosY) * math.cos(
                rx) + centerPosY
            str_out = str_out + ' ' + str(newRestPosX)
            str_out = str_out + ' ' + str(newRestPosY)
            str_out = str_out + ' ' + str(rest_pos[i][2])
            str_out = str_out + ' ' + str(qx)
            str_out = str_out + ' ' + str(qy)
            str_out = str_out + ' ' + str(qz)
            str_out = str_out + ' ' + str(w)
        self.rotAngle = self.rotAngle + self.finger_rotation_step * math.pi / 180
        self.MecaObjectfinger.findData('rest_position').value = str_out
        return str_out
    def resetskinshape(self, str_original_skin_pos):
        self.MecaObjectskin.findData('position').value = str_original_skin_pos

    def resetmarkershape(self, str_original_marker_pos):
        self.MecaObjectmarker.findData('position').value = str_original_marker_pos

    def resetfingerpos(self, str_original_finger_pos):
        self.MecaObjectfinger.findData('position').value = str_original_finger_pos

    def initGraph(self, node):

        self.node = node
        self.a = 0
        self.dt = self.node.findData('dt').value

        # Measurement setting
        # starting time
        self.init_indention_time = round(0.05 * 100, 0)

        # finish indention and start moving to the next one
        self.init_restriction_time = round(self.init_indention_time + 3 * 100 + 0.4 * 100, 0)
        # number of time moving up finger
        self.Number_measured_point = 0
        # number of time rotating finger
        self.Number_rotation_step = 0
        # Euler angle for each time finger rotation
        self.finger_rotation_step = 10
        # moving distance in z direction
        self.vertical_step = 5
        # Measuring time
        self.indention_time = round(self.init_restriction_time - self.init_indention_time, 0)
        self.shape_recovering_time = 0.2 * 100
        # Distance to contact = contact distance in localmindistance + gap from DOF to the contact surface of finger
        self.dist_contact = 15
        # pre-gap before contact
        self.pre_gap = 2.0
        self.moving_time = 0
        self.centerPosX = 0
        self.centerPosY = 0
        self.centerPosZ = 0

        self.force = [0, 0, 0]
        # Skin node:
        self.skin = self.node.getChild('skin')
        self.MecaObjectskin = self.skin.getObject('DOFs')
        self.topology_container = self.skin.getObject('topology')
        self.original_skin_pos = self.topology_container.position
        self.skinVisual = self.skin.getChild('Visual')
        self.skinColli = self.skin.getChild('SkinCollision')
        self.Mecaskincolli = self.skinColli.getObject('collisMech')
        self.outer_indices = []
        self.outer_indices2 = []
        for i in outer_skin_indices:
            if int(self.original_skin_pos[i][2]) > 20 and int(self.original_skin_pos[i][2]) < 240:
                self.outer_indices.extend([i])
        for i in outer_skin_indices:
            if int(self.original_skin_pos[i][2]) > 0.5 and int(self.original_skin_pos[i][2]) < 259.5:
                self.outer_indices2.extend([i])
        print(self.outer_indices)
        print(len(self.outer_indices))
        self.measuring_pos_skin = self.MecaObjectskin.findData('position').value[self.outer_indices[0]]
        self.rotAngle = math.atan2(self.measuring_pos_skin[1],self.measuring_pos_skin[0])
        # Finger node:
        self.fingerNode = self.node.getChild('finger')
        self.MecaObjectfinger = self.fingerNode.getObject('tetras')

        self.fingercolli = self.fingerNode.getChild('collisionFinger')
        self.Mecafingercolli = self.fingercolli.getObject('collisMech')
        self.str_original_skin_pos = ' '
        for i in xrange(0, len(self.original_skin_pos)):
            self.str_original_skin_pos = self.str_original_skin_pos + ' ' + str(self.original_skin_pos[i][0])
            self.str_original_skin_pos = self.str_original_skin_pos + ' ' + str(self.original_skin_pos[i][1])
            self.str_original_skin_pos = self.str_original_skin_pos + ' ' + str(self.original_skin_pos[i][2])

        self.str_init_finger_pos = ' '
        self.str_init_finger_pos = self.str_init_finger_pos + ' ' + str(
            self.original_skin_pos[self.outer_indices[0]][0]+17*math.cos(self.rotAngle))
        self.str_init_finger_pos = self.str_init_finger_pos + ' ' + str(
            self.original_skin_pos[self.outer_indices[0]][1]+17*math.sin(self.rotAngle))
        self.str_init_finger_pos = self.str_init_finger_pos + ' ' + str(
            self.original_skin_pos[self.outer_indices[0]][2])

        self.MecaObjectfinger.findData('translation').value = self.str_init_finger_pos

        self.str_init_finger_rot = ' '
        self.str_init_finger_rot = self.str_init_finger_rot + ' ' + str(0)
        self.str_init_finger_rot = self.str_init_finger_rot + ' ' + str(90)
        self.str_init_finger_rot = self.str_init_finger_rot + ' ' + str(self.rotAngle*180/math.pi)

        self.MecaObjectfinger.findData('rotation').value = self.str_init_finger_rot
        # Marker node:
        self.markerNode = self.node.getChild('markers')
        self.markerVisual = self.markerNode.getChild('Visual')
        self.marker_topology_container =self.markerNode.getObject('topology')
        self.MecaObjectmarker = self.markerNode.getObject('DOFs-marker')
        self.original_marker_pos = self.marker_topology_container.position
        self.str_original_marker_pos = ' '
        for i in xrange(0, len(self.original_marker_pos)):
            self.str_original_marker_pos = self.str_original_marker_pos + ' ' + str(self.original_marker_pos[i][0])
            self.str_original_marker_pos = self.str_original_marker_pos + ' ' + str(self.original_marker_pos[i][1])
            self.str_original_marker_pos = self.str_original_marker_pos + ' ' + str(self.original_marker_pos[i][2])

        # directory = "contact-info"
        # parent_dir = "/respitory adress"
        # filePath = os.path.join(parent_dir,directory)
        # os.mkdir(filePath)
        # for i in self.outer_indices:
        #     directory = "contact-node-"+str(i)
        #     parent_dir = "respitory adress"
        #     filePath = os.path.join(parent_dir, directory)
        #     os.mkdir(filePath)
        #     for j in range(30):
        #         directory = "contact-depth-" + str(float(j+1))
        #         parent_dir = "respitory adress/"+"contact-node-" + str(i)
        #         filePath = os.path.join(parent_dir, directory)
        #         os.mkdir(filePath)

    def onKeyPressed(self, c):
        self.Velocity_measure = -10

    def onBeginAnimationStep(self, deltaTime):

        self.MecaObjectfinger = self.fingerNode.getObject('tetras')
        # Indention velocity
        self.Velocity_measure = -10
        Travel_measuring_step = self.dt * self.Velocity_measure   ## cu 1 dt step luong dich chuyen la Travel_step
        self.approach_time = self.pre_gap*100/abs(self.Velocity_measure)
        # time_step = float(format(self.node.getRoot().time,".2f"))*100
        time_step = round(self.node.getRoot().time*100,0)
        self.original_finger_pos = self.MecaObjectfinger.findData('reset_position').value
        self.init_reset_finger = 0
        h_layer = [0, 25.0, 30.0, 35.0]
        if self.Number_measured_point == len(self.outer_indices):
            self.node.getRootContext().animate = False
        else:
            if (0 <self.MecaObjectskin.findData('reset_position').value[self.outer_indices[self.Number_measured_point]][2] <= h_layer[3]) or (260-h_layer[3] <self.MecaObjectskin.findData('reset_position').value[self.outer_indices[self.Number_measured_point]][2] <= 260):
                for j in range(len(h_layer)-1):
                    if (self.MecaObjectskin.findData('reset_position').value[self.outer_indices[self.Number_measured_point]][2] > h_layer[j] and self.MecaObjectskin.findData('reset_position').value[self.outer_indices[self.Number_measured_point]][2] <= h_layer[j+1])\
                            or (self.MecaObjectskin.findData('reset_position').value[self.outer_indices[self.Number_measured_point]][2] > 260-h_layer[j+1] and self.MecaObjectskin.findData('reset_position').value[self.outer_indices[self.Number_measured_point]][2] <= 260-h_layer[j]):
                        if time_step >= round(self.init_indention_time, 0) and time_step <= round(self.init_restriction_time - (3-j) * 100,0):
                            self.indention(Travel_measuring_step * math.cos(self.rotAngle),
                                           Travel_measuring_step * math.sin(self.rotAngle))
                            if time_step > round(self.init_indention_time + self.approach_time, 0) and time_step <= round(self.init_restriction_time - (3-j) * 100, 0):
                                self.skinVisual.getObject('skin_exporter').findData('enable').value = 1
                                self.markerVisual.getObject('marker_exporter').findData('enable').value = 1
                                self.moving_time = self.moving_time + 1
                                self.skinVisual.getObject('skin_exporter').filename = \
                                    "respitory adress/skin"
                                self.markerVisual.getObject('marker_exporter').filename = \
                                    "respitory adress/marker"
                                for i in range(1, 31):
                                    if self.moving_time / float(self.skinVisual.getObject('skin_exporter').exportEveryNumberOfSteps) == i:
                                        current_pos_skin = []
                                        current_distributed_force = []
                                        current_velocity = []
                                        skin_pos = self.MecaObjectskin.position
                                        skin_force = self.MecaObjectskin.force
                                        skin_vel = self.MecaObjectskin.velocity
                                        for i in self.outer_indices2:
                                            current_pos_skin += [skin_pos[i]]
                                            current_distributed_force += [skin_force[i]]
                                            current_velocity += [skin_vel[i]]

                                        pos_x = []
                                        pos_y = []
                                        pos_z = []
                                        force_x = []
                                        force_y = []
                                        force_z = []
                                        vel_x = []
                                        vel_y = []
                                        vel_z = []

                                        for i in range(len(current_pos_skin)):
                                            pos_x.extend([current_pos_skin[i][0]])
                                            pos_y.extend([current_pos_skin[i][1]])
                                            pos_z.extend([current_pos_skin[i][2]])
                                        for i in range(len(current_distributed_force)):
                                            force_x.extend([current_distributed_force[i][0] * 0.1])
                                            force_y.extend([current_distributed_force[i][1] * 0.1])
                                            force_z.extend([current_distributed_force[i][2] * 0.1])
                                        for i in range(len(current_velocity)):
                                            vel_x.extend([current_velocity[i][0]])
                                            vel_y.extend([current_velocity[i][1]])
                                            vel_z.extend([current_velocity[i][2]])
                                        data_to_write = [self.outer_indices2] + [pos_x] + [pos_y] + [pos_z] + [force_x] + [force_y] + [force_z] + [vel_x] + [vel_y] + [vel_z]

                                        bufsize = 0
                                        filePath = "respitory adress" + "contact-node-" \
                                                   + str(self.outer_indices[self.Number_measured_point]) + "/contact-depth-" \
                                                   + str(abs(self.moving_time * Travel_measuring_step)) + "/data.csv"
                                        try:
                                            os.remove(filePath)
                                        except:
                                            print("Error while deleting file ", filePath)
                                        file_open = open(filePath, "w",
                                                         buffering=bufsize)
                                        header = ["Node index", "Posx (mm)", "Posy (mm)", "Posz (mm)", "Fx (N)", "Fy (N)",
                                                  "Fz (N)", "Vx (mm/s)", "Vy (mm/s)", "Vz (mm/s)"]
                                        writer = csv.writer(file_open)
                                        writer.writerow(header)
                                        print('IN BOUNDARY REGION: contact depth = ', self.moving_time*Travel_measuring_step)
                                        for i in range(len(self.outer_indices2)):
                                            writer.writerow(zip(*data_to_write)[i])
                            else:
                                self.moving_time = 0
                                self.skinVisual.getObject('skin_exporter').findData('enable').value = False
                                self.markerVisual.getObject('marker_exporter').findData('enable').value = False
                        elif time_step > round(self.init_restriction_time - (3-j) * 100, 0):
                            self.skinVisual.getObject('skin_exporter').findData('enable').value = False
                            self.markerVisual.getObject('marker_exporter').findData('enable').value = False
                            self.init_indention_time = round(time_step + self.shape_recovering_time,0)
                            self.init_restriction_time = round(self.init_indention_time + self.indention_time,0)
                            self.Number_measured_point += 1
                            print("Current contact point: ", self.outer_indices[self.Number_measured_point])
                            print("Z = ", self.MecaObjectskin.findData('reset_position').value[self.outer_indices[self.Number_measured_point]][2])
                            self.measuring_pos_skin = self.MecaObjectskin.findData('reset_position').value[
                                self.outer_indices[self.Number_measured_point]]
                            self.movenext(self.measuring_pos_skin[0], self.measuring_pos_skin[1], self.measuring_pos_skin[2])
                            self.resetskinshape(self.str_original_skin_pos)
                            self.resetmarkershape(self.str_original_marker_pos)
                            self.moving_time = 0.0
            else:
                if time_step >= round(self.init_indention_time, 0) and time_step <= round(self.init_restriction_time):
                    self.indention(Travel_measuring_step * math.cos(self.rotAngle),
                                   Travel_measuring_step * math.sin(self.rotAngle))
                    if time_step > round(self.init_indention_time + self.approach_time, 0) and time_step <= round(self.init_restriction_time):
                        self.skinVisual.getObject('skin_exporter').findData('enable').value = 1
                        self.markerVisual.getObject('marker_exporter').findData('enable').value = 1
                        self.moving_time = self.moving_time + 1.0
                        self.skinVisual.getObject('skin_exporter').filename = \
                            "respitory adress/skin"
                        self.markerVisual.getObject('marker_exporter').filename = \
                            "respitory adress/marker"
                        for i in range(1,31):
                            if self.moving_time / float(self.skinVisual.getObject('skin_exporter').exportEveryNumberOfSteps) == i:
                                current_pos_skin = []
                                current_distributed_force = []
                                current_velocity = []
                                skin_pos = self.MecaObjectskin.position
                                skin_force = self.MecaObjectskin.force
                                skin_vel = self.MecaObjectskin.velocity
                                for i in self.outer_indices2:
                                    current_pos_skin += [skin_pos[i]]
                                    current_distributed_force += [skin_force[i]]
                                    current_velocity += [skin_vel[i]]
                                pos_x = []
                                pos_y = []
                                pos_z = []
                                force_x = []
                                force_y = []
                                force_z = []
                                vel_x = []
                                vel_y = []
                                vel_z = []

                                for i in range(len(current_pos_skin)):
                                    pos_x.extend([current_pos_skin[i][0]])
                                    pos_y.extend([current_pos_skin[i][1]])
                                    pos_z.extend([current_pos_skin[i][2]])
                                for i in range(len(current_distributed_force)):
                                    force_x.extend([current_distributed_force[i][0]*0.1])
                                    force_y.extend([current_distributed_force[i][1]*0.1])
                                    force_z.extend([current_distributed_force[i][2]*0.1])
                                for i in range(len(current_velocity)):
                                    vel_x.extend([current_velocity[i][0]])
                                    vel_y.extend([current_velocity[i][1]])
                                    vel_z.extend([current_velocity[i][2]])
                                data_to_write = [self.outer_indices2] + [pos_x] + [pos_y] + [pos_z] + [force_x] + [force_y] + [force_z] + [vel_x]+ [vel_y]+ [vel_z]

                                bufsize = 0
                                filePath = "respitory adress" + "contact-node-" \
                                           + str(self.outer_indices[self.Number_measured_point]) + "/contact-depth-"\
                                           + str(abs(self.moving_time * Travel_measuring_step)) + "/data.csv"
                                try:
                                    os.remove(filePath)
                                except:
                                    print("Error while deleting file ", filePath)
                                file_open = open(filePath , "w",
                                              buffering=bufsize)
                                header = ["Node index", "Posx (mm)", "Posy (mm)", "Posz (mm)", "Fx (N)", "Fy (N)", "Fz (N)", "Vx (mm/s)", "Vy (mm/s)", "Vz (mm/s)"]
                                writer = csv.writer(file_open)
                                writer.writerow(header)
                                print('IN MIDDLE REGION: contact depth = ', self.moving_time * Travel_measuring_step)
                                for i in range(len(self.outer_indices2)):
                                    writer.writerow(zip(*data_to_write)[i])

                    else:
                        self.moving_time = 0.0
                        self.skinVisual.getObject('skin_exporter').findData('enable').value = False
                        self.markerVisual.getObject('marker_exporter').findData('enable').value = False
                elif time_step > round(self.init_restriction_time, 0):
                    self.skinVisual.getObject('skin_exporter').findData('enable').value = False
                    self.markerVisual.getObject('marker_exporter').findData('enable').value = False
                    self.init_indention_time = round(time_step + self.shape_recovering_time,0)
                    self.init_restriction_time = round(self.init_indention_time + self.indention_time,0)
                    self.Number_measured_point += 1
                    print("Current contact point: ", self.outer_indices[self.Number_measured_point])
                    print("Z = ", self.MecaObjectskin.findData('reset_position').value[
                        self.outer_indices[self.Number_measured_point]][2])
                    self.measuring_pos_skin = self.MecaObjectskin.findData('reset_position').value[
                        self.outer_indices[self.Number_measured_point]]
                    self.movenext(self.measuring_pos_skin[0], self.measuring_pos_skin[1], self.measuring_pos_skin[2])
                    self.resetskinshape(self.str_original_skin_pos)
                    self.resetmarkershape(self.str_original_marker_pos)
                    self.moving_time = 0

