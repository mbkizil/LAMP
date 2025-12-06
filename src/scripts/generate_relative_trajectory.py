import numpy as np
from scipy.spatial.transform import Rotation # Sadece 'look_at' için
import random # Jitter için eklendi
import re # (YENİ) Rotation parser için eklendi
import argparse # (YENİ) Main için eklendi
import sys # (YENİ) Main için eklendi

# --- Gerekli Yardımcılar (Eski kodunuzda varsayıldı) ---

# Z_OFFSET: Kameranın objeye olan varsayılan 'dinlenme' mesafesi (pozitif olmalı)
# Lütfen bu sabiti kendi kodunuzdaki değeriyle güncelleyin.
SPACE_ORIGIN = 256.0
SPACE_SCALE = 256.0
MIN_VAL = 0
MAX_VAL = 512
Z_OFFSET = 100/256

def lerp(a, b, t):
# ... (Mevcut kod - Değişiklik yok) ...
    """Lineer enterpolasyon: a'dan b'ye t kadar git."""
    return a * (1.0 - t) + b * t

# (YENİ) 3D Vektör Lerp (Rotation Track için eklendi)
def lerp_3d(a_tuple, b_tuple, t):
# ... (Mevcut kod - Değişiklik yok) ...
    """İki (x,y,z) tuple'ı arasında lineer enterpolasyon."""
    return (
        lerp(a_tuple[0], b_tuple[0], t),
        lerp(a_tuple[1], b_tuple[1], t),
        lerp(a_tuple[2], b_tuple[2], t)
    )

# (YENİ) Vektör Normalleştirme (Rotation Track için eklendi)
def normalize_vector(v_tuple):
# ... (Mevcut kod - Değişiklik yok) ...
    """Bir (x,y,z) tuple'ını normalize eder, (vektör, büyüklük) döndürür."""
    v = np.array(v_tuple)
    mag = np.linalg.norm(v)
    if mag == 0:
        return (0.0, 0.0, 0.0), 0.0
    v_norm = v / mag
    return (v_norm[0], v_norm[1], v_norm[2]), mag


# (YENİ) Easing (Hızlanma/Yavaşlama) fonksiyonları
def apply_easing(t, style):
# ... (Mevcut kod - Değişiklik yok) ...
    """
    Verilen 't' (0.0-1.0) değerine DSL easing stilini uygular.
    """
    if style == 'ease_in':
        # (Quad In)
        return t * t
    elif style == 'ease_out':
        # (Quad Out)
        return t * (2.0 - t)
    elif style == 'ease_in_out':
        # (Quad In-Out)
        if t < 0.5:
            return 2.0 * t * t
        return -1.0 + (4.0 - 2.0 * t) * t
    elif style == 'ease_out_in':
        if t < 0.5:
            return 0.5 * (1.0 - (1.0 - 2.0 * t) * (1.0 - 2.0 * t))
        return 0.5 + 0.5 * (2.0 * t - 1.0) * (2.0 * t - 1.0)
    else: # 'linear' veya bilinmeyen
        return t

# (YENİ) Jitter (Titreme) fonksiyonu
def get_jitter_offset(style):
# ... (Mevcut kod - Değişiklik yok) ...
    """
    DSL jitter stiline göre rastgele bir pozisyon ofseti döndürür.
    """
    if style == 'low':
        mag = 0.002 # Düşük titreme büyüklüğü
    elif style == 'high':
        mag = 0.005 # Yüksek titreme büyüklüğü
    else: # 'none'
        return (0.0, 0.0, 0.0)
    
    return (random.uniform(-mag, mag), 
            random.uniform(-mag, mag), 
            random.uniform(-mag, mag))



def calculate_look_at_quaternion(cam_pos, obj_pos, fixed_axis=None, roll_angle_rad=0.0):
    """
    Kameranın 'obj_pos' noktasına bakması için gerekli quaternion'u hesaplar.
    GÜNCELLENDİ: Artık 'roll_angle_rad' parametresi alıyor.
    """
    # Bakış vektörünü (direction) al
    look_at_vector = (obj_pos[0] - cam_pos[0], 
                      obj_pos[1] - cam_pos[1], 
                      obj_pos[2] - cam_pos[2])

    # Vektörü normalize et
    v_norm, magnitude = normalize_vector(look_at_vector)
    
    if magnitude == 0.0:
        # Obje kameranın içinde, identity quaternion döndür
        return 1.0, 0.0, 0.0, 0.0

    vx, vy, vz = v_norm
    
    # Euler Açılarını Hesapla (Y up, -Z forward)
    yaw_angle = np.arctan2(vx, -vz)
    pitch_angle = np.asin(vy)
    # Roll açısı artık parametre olarak geliyor
    roll_angle = roll_angle_rad
    
    # 'fixed_axis' kısıtlamasını uygula
    if fixed_axis == 'x': # Pitch'i kilitle (yukarı/aşağı bakma)
        pitch_angle = 0.0
    elif fixed_axis == 'y': # Yaw'ı kilitle (sağa/sola bakma)
        yaw_angle = 0.0
    # 'z' (roll) artık kilitli değil, parametreye bağlı
    
    try:
        r = Rotation.from_euler('yxz', [yaw_angle, pitch_angle, roll_angle])
        q_scipy = r.as_quat()
    except Exception as e:
        print(f"UYARI: Quaternion hesaplama hatası: {e}", file=sys.stderr)
        q_scipy = np.array([0.0, 0.0, 0.0, 1.0]) # Hata durumunda Identity

    # Sizin formatınıza dönüştür: [rw, rz, rx, ry]
    rw = q_scipy[3] # w
    rx = q_scipy[0] # x
    ry = q_scipy[1] # y
    rz = q_scipy[2] # z

    return rw, rz, rx, ry

def interpolate_object_path(obj_points_float):
    """
    Verilen 5 *float* yörünge noktasını 21 *float* frame'lik
    bir yola enterpole eder.
    """
    interpolated_points = []
    interpolated_points.append(obj_points_float[0]) # Frame 0
    for i in range(4):
        p_start = obj_points_float[i]
        p_end = obj_points_float[i+1]
        for frame_step in range(1, 6): # 1, 2, 3, 4, 5
            t = frame_step / 5.0
            interpolated_points.append(lerp_3d(p_start, p_end, t))
    return interpolated_points


def parse_object_trajectory(obj_line):
    """
    (GÜNCELLENDİ) Obje satırını (int) okur.
    Virgül veya '|' ile ayrılmış olabilir.
    Eğer uzunluk Nx3 ise, eşit aralıklarla 5 frame alır ve döndürür.
    """
    try:
        # Ayırıcı: | , veya boşluk olabilir
        values = [int(v) for v in re.split(r'[| ,]+', obj_line.strip()) if v]
        
        if len(values) == 0:
            return None
        
        # 3'e bölünebilir olmalı
        if len(values) % 3 != 0:
            return None

        num_frames = len(values) // 3
        arr = np.array(values).reshape(num_frames, 3)

        # Eğer 5'ten az frame varsa hepsini döndür
        if num_frames <= 5:
            return arr

        # Eşit aralıklarla 5 frame seç
        indices = np.linspace(0, num_frames - 1, 5, dtype=int)
        sampled = arr[indices]
        return sampled

    except Exception as e:
        sys.stderr.write(f"UYARI: Obje yörünge ayrıştırma hatası: {e}\n")
        sys.exit(1)
        return None
def process_object_trajectory(obj_points_int):
    """
    5 noktalık tamsayı yörüngeyi alır, [-1, 1] aralığına normalize eder
    ve Z-ofsetini uygular. 5 noktalık *float* yörünge döndürür.
    """
    processed_points = []
    for p in obj_points_int:
        norm_x = (p[0] - SPACE_ORIGIN) / SPACE_SCALE
        norm_y = (p[1] - SPACE_ORIGIN) / SPACE_SCALE
        norm_z = (p[2] - SPACE_ORIGIN) / SPACE_SCALE
        
        # Obje, kameranın önünde (-Z) başlasın diye
        norm_z -= Z_OFFSET 
        
        processed_points.append( (norm_x, norm_y, norm_z) )
    return processed_points

def denormalize_value(v):
    """
    [PLACEHOLDER] [-1, 1] float'ı [0, 512] int'e çevirir.
    Kodunuzda bu fonksiyonun zaten olduğunu varsayıyoruz.
    """
    # (v + 1.0) * 0.5 * 512.0
    val_01 = (v + 1.0) * 0.5
    val_mapped = val_01 * 512.0
    return int(np.round(val_mapped))

def handle_unknown_tag(obj_points_float):
    """
    [PLACEHOLDER] Bilinmeyen etiket için varsayılan (sabit) yörünge.
    Kodunuzda bu fonksiyonun zaten olduğunu varsayıyoruz.
    """
    # print("Uyarı: 'handle_unknown_tag' placeholder kullanılıyor.")
    identity_frame = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return [identity_frame] * 21 # 21 frame'lik sabit yörünge


def tail_track(obj_points_float, 
               follow_style='hard',      # DSL: follow_style_hard
               follow_axis=[],         # DSL: ['x', 'y'] (boş liste 'full' demektir)
               look_at=True,           # GÜNCELLENDİ (v11): Artık default True
               dolly=(None, 0.0),      # DSL: ('in', 0.2) veya ('out', 0.3)
               amplitude=(1.0, 1.0, 1.0), # DSL: (amp_x, amp_y, amp_z)
               mirror_axis=[],          # DSL: ['x', 'y']
               lead=False,              # GÜNCELLENDİ (v11.12): Artık kullanılıyor
               # --- COMMON MODIFIERS ---
               easing='linear',        # DSL: ease_in
               dutch_angle_deg=0,      # DSL: dutch_15
               jitter_style='none',    # DSL: jitter_low
               # --- YENİ (v11.5) ---
               vertical_angle='none',
               framing_offset='none'
               ):
    """
    DSL (Bölüm 5) ile uyumlu, gelişmiş 'tail_tracking' fonksiyonu.
    GÜNCELLENDİ (v11.12): 'lead' mantığı eklendi.
    1. Kamera, objenin ilk hareket yönüne Z_OFFSET kadar önden başlar (Pozisyon).
    2. 'lead=True', 'look_at=False' ('dont_look') etiketini geçersiz kılar
       ve kamerayı objeye bakmaya zorlar (Rotasyon).
    """
    
    # 1. Obje yolunu al (0-20 indeksli)
    obj_path_21 = interpolate_object_path(obj_points_float)
    
    # 2. Objenin başlangıç pozisyonunu al
    obj_pos_0 = obj_path_21[0]
    
    camera_frames = [] # Boş liste ile başla
    
    # 3. Frame 0 (Identity) kaldırıldı.
    
    # 4. Damping (Gecikme) için ayarlar
    damping_map = {'hard': 1.0, 'soft': 0.5, 'lazy': 0.1}
    damping_factor = damping_map.get(follow_style, 1.0) 
    
    # prev_cam_pos, damping için bir önceki frame'in 'damped delta'sını tutar
    prev_cam_pos = (0.0, 0.0, 0.0) 
    num_frames = 20.0 # float bölme için (0'dan 20'ye 21 frame)
    
    # 5. Dolly (mesafe) hedeflerini belirle
    dolly_dir, dolly_amount_parsed = dolly
    dolly_amount = 0.0
    if dolly_dir == 'in':
        dolly_amount = dolly_amount_parsed
    elif dolly_dir == 'out':
        dolly_amount = -dolly_amount_parsed

    start_distance = Z_OFFSET
    end_distance = Z_OFFSET * (1.0 - dolly_amount)
    
    # YENİ (v11.5): Pozisyonel Ofsetleri Hesapla
    vertical_offset_amount = 50.0 / 256.0 
    vertical_offset_y = 0.0
    if vertical_angle == 'aerial':
        vertical_offset_y = vertical_offset_amount
    elif vertical_angle == 'low_angle':
        vertical_offset_y = -vertical_offset_amount
        
    framing_offset_amount = 50.0 / 256.0 
    framing_offset_x = 0.0
    if framing_offset == 'object_left':
        framing_offset_x = framing_offset_amount # Kamerayı Sağa al (Obje Solda)
    elif framing_offset == 'object_right':
        framing_offset_x = -framing_offset_amount # Kamerayı Sola al (Obje Sağda)
        
    has_framing_offset = (framing_offset_x != 0.0)
    
    # YENİ (v11.12): 'lead' ofsetini hesapla
    lead_offset_vec = (0.0, 0.0, 0.0)
    if lead:
        # "İlk yönü" ilk 5 frame'in (ilk segment) yönü olarak al
        obj_pos_5 = obj_path_21[5]
        initial_vec = (obj_pos_5[0] - obj_pos_0[0], 
                       obj_pos_5[1] - obj_pos_0[1], 
                       obj_pos_5[2] - obj_pos_0[2])
        
        (initial_dir_vec, initial_mag) = normalize_vector(initial_vec)
        
        # Sadece obje hareket ediyorsa lead uygula (mag > 0)
        if initial_mag > 1e-6:
            lead_offset_vec = (initial_dir_vec[0] * Z_OFFSET*0.7, 
                             initial_dir_vec[1] * Z_OFFSET*0.7, 
                             initial_dir_vec[2] * Z_OFFSET*0.7 - Z_OFFSET)
    
    
    # 6. GÜNCELLENDİ: Döngü 0'dan 20'ye (dahil) başlıyor (Toplam 21 frame)
    for i in range(0, 21): 
        obj_pos_i = obj_path_21[i] # Objenin o anki gerçek konumu
        
        # Yörüngedeki ilerleme (0.0'dan 1.0'a)
        t_linear = i / num_frames
        
        # Easing uygula
        t_eased = apply_easing(t_linear, easing)
        
        # --- KAMERA POZİSYON HESAPLAMA PİPELINE'I ---
        
        # --- A. HEDEF DELTA HESAPLAMASI ---
        
        # 1. Objenin o anki deltasını ve genliğini hesapla
        target_tx = (obj_pos_i[0] - obj_pos_0[0]) * amplitude[0]
        target_ty = (obj_pos_i[1] - obj_pos_0[1]) * amplitude[1]
        target_tz = (obj_pos_i[2] - obj_pos_0[2]) * amplitude[2]
        
        # 2. Mirror (Aynalama) uygula
        if 'x' in mirror_axis:
            target_tx = -target_tx
        if 'y' in mirror_axis:
            target_ty = -target_ty

        # 3. Follow Axis (Takip Ekseni) kısıtlamasını uygula
        final_target_tx = target_tx
        final_target_ty = target_ty
        final_target_tz = target_tz
        
        if follow_axis:
            if 'x' not in follow_axis: final_target_tx = 0.0
            if 'y' not in follow_axis: final_target_ty = 0.0
            if 'z' not in follow_axis: final_target_tz = 0.0
        
        # --- B. POZİSYON HESAPLAMASI (LERP, DOLLY, OFFSET) ---

        # 1. X ve Y'yi Damping (gecikme) ile hesapla
        tx_damped = lerp(prev_cam_pos[0], final_target_tx, damping_factor)
        ty_damped = lerp(prev_cam_pos[1], final_target_ty, damping_factor)

        # 2. Z'yi hesapla (Dolly veya Normal Takip)
        tz_damped = 0.0
        if dolly_amount != 0.0:
            current_distance = lerp(start_distance, end_distance, t_eased) # Easing'li 't_eased'
            tz_damped = obj_pos_i[2] + current_distance
        else:
            tz_damped = lerp(prev_cam_pos[2], final_target_tz, damping_factor)
        
        # Damping için bir sonraki frame'in pozisyonunu (ofsetsiz) sakla
        prev_cam_pos = (tx_damped, ty_damped, tz_damped)
        
        # 3. YENİ (v11.12): Tüm Pozisyonel Ofsetleri Damped Pozisyona Ekle
        cam_pos_base = (
            tx_damped + framing_offset_x + lead_offset_vec[0], # Lead X
            ty_damped + vertical_offset_y + lead_offset_vec[1], # Lead Y
            tz_damped + lead_offset_vec[2]                   # Lead Z
        )

        # 4. Jitter (Titreme) uygula
        if jitter_style != 'none':
            j_dx, j_dy, j_dz = get_jitter_offset(jitter_style)
            cam_pos = (cam_pos_base[0] + j_dx, 
                       cam_pos_base[1] + j_dy, 
                       cam_pos_base[2] + j_dz)
        else:
            cam_pos = cam_pos_base
        
        # --- HESAPLAMA PİPELINE'I BİTTİ ---
        
        
        # Adım 7: 'look_at' ve 'dutch_angle' rotasyonunu hesapla
        
        obj_target_pos = obj_pos_i # Varsayılan hedef
        
        # 7a. 'look_at' hesapla
        
        # GÜNCELLENDİ (v11.12): 'lead' etiketi, 'look_at=True' mantığını zorunlu kılar.
        if look_at or lead: 
            
            # YENİ (v11.5): Pan-to-center hedef enterpolasyonu
            obj_target_pos_end = obj_pos_i # t=1 hedefi
            
            if has_framing_offset:
                # t=0 hedefi (pan-ofsetli)
                obj_target_pos_start = (
                    obj_pos_i[0] + framing_offset_x,
                    obj_pos_i[1],
                    obj_pos_i[2]
                )
                # t=0 -> start, t=1 -> end
                obj_target_pos = lerp_3d(obj_target_pos_start, obj_target_pos_end, t_eased)
            
            rw_look, rz_look, rx_look, ry_look = calculate_look_at_quaternion(
                cam_pos, 
                obj_target_pos, # Enterpole edilmiş hedef
                fixed_axis=None
            ) 
            q_look = Rotation.from_quat([rx_look, ry_look, rz_look, rw_look])
        else:
            # GÜNCELLENDİ (v11.6): 'dont_look' modu
            # (Sadece 'look_at=False' VE 'lead=False' ise buraya gelir)
            
            # 'vertical_angle' varsa tilt yap, ama pan yapma (yaw'ı kilitle).
            # 'cam_pos' zaten dikey ofseti içeriyor.
            rw_look, rz_look, rx_look, ry_look = calculate_look_at_quaternion(
                cam_pos, 
                obj_pos_i, # Hedef olarak objeyi kullan
                fixed_axis='y' # <-- AMA YAW (PAN) EKSENİNİ KİLİTLE
            ) 
            q_look = Rotation.from_quat([rx_look, ry_look, rz_look, rw_look])
            q_look = Rotation.from_quat([0.0, 0.0, 0.0, 1.0]) # Identity Quaternion
        
        q_final = q_look # Başlangıç

        # 7b. 'dutch_angle' uygula
        if dutch_angle_deg != 0:
            # GÜNCELLENDİ: Easing'i Dutch Angle'a da uygula
            current_dutch_deg = lerp(0.0, dutch_angle_deg, t_eased)
            
            q_roll = Rotation.from_euler('z', current_dutch_deg, degrees=True)
            q_final = q_look * q_roll
        
        # 7c. Sonucu (w, z, x, -ry) formatına geri çevir
        final_quat_xyzw = q_final.as_quat() # (x, y, z, w)
        rw, rx, ry, rz = final_quat_xyzw[3], final_quat_xyzw[0], final_quat_xyzw[1], final_quat_xyzw[2]
        
        # Adım 8: Frame'i ekle
        tx_final, ty_final, tz_final = cam_pos
        camera_frames.append( (rw, rz, rx, -ry, tx_final, ty_final, tz_final) )
        
        # 'prev_cam_pos' zaten döngünün başında güncellendi
        
    return camera_frames

# =========================================================================
# (YENİ) rotation_track Fonksiyonu ve Mantığı
# =========================================================================

def rotation_track(
    obj_points_float,
    rotation_axis='full',       # DSL: pan_only, tilt_only
    local_dolly=(None, 0.0),    # DSL: ('in', 0.3)
    world_move_1={},            # DSL: {'x': 0.2, 'y': -0.1}
    world_move_2={},            # DSL: {'z': 0.4}
    look_offset=(0.0, 0.0),     # DSL: (offset_x, offset_y)
    # --- COMMON MODIFIERS ---
    easing='linear',
    dutch_angle_deg=0,
    jitter_style='none',
    # --- YENİ (v11) ---
    vertical_angle='none',       # 'aerial', 'low_angle', veya 'none'
    framing_offset='none'        # 'object_left', 'object_right'
    ):
    """
    DSL (Bölüm 6) ile uyumlu, gelişmiş 'rotation_track' fonksiyonu.
    GÜNCELLENDİ (v11.5): 'vertical_angle' ve 'framing_offset' artık
    kameranın pozisyonunu (tx, ty, tz) etkiliyor VE Frame 0 'identity' değil.
    """
    
    # 1. Obje yolunu al (0-20 indeksli)
    obj_path_21 = interpolate_object_path(obj_points_float)
    camera_frames = [] # Boş liste ile başla
    look_offset = (-look_offset[0]/4, look_offset[1]/4) # DSL offset küçültme
    
    # 2. Frame 0 (Identity) kaldırıldı.
    
    num_frames = 20.0 # float bölme için (0'dan 20'ye 21 frame)
    start_pos = (0.0, 0.0, 0.0)
    
    # 3. DSL parametrelerini ayarla
    fixed_axis_map = {'pan_only': 'x', 'tilt_only': 'y', 'full': None}
    fixed_axis = fixed_axis_map.get(rotation_axis) # 'calculate_look_at' için

    # --- YENİ v11.5: Ofsetleri Hesapla (Pozisyon için) ---
    vertical_offset_amount = 50.0 / 256.0 
    vertical_offset_y = 0.0
    if vertical_angle == 'aerial':
        vertical_offset_y = vertical_offset_amount
    elif vertical_angle == 'low_angle':
        vertical_offset_y = -vertical_offset_amount
        
    framing_offset_amount = 50.0 / 256.0 
    framing_offset_x = 0.0
    if framing_offset == 'object_left':
        framing_offset_x = framing_offset_amount # Kamerayı Sağa al (Obje Solda)
    elif framing_offset == 'object_right':
        framing_offset_x = -framing_offset_amount # Kamerayı Sola al (Obje Sağda)
        
    has_framing_offset = (framing_offset_x != 0.0)
    # --- Bitiş v11.5 ---

    # World move hedeflerini vektörlere çevir (TEMİZ, v11 OFSETSİZ)
    target_1_vec_clean = (world_move_1.get('x', 0.0), world_move_1.get('y', 0.0), -world_move_1.get('z', 0.0))
    target_2_vec_clean = (world_move_2.get('x', 0.0), world_move_2.get('y', 0.0), -world_move_2.get('z', 0.0))
    
    target_2_combined_clean = (target_1_vec_clean[0] + target_2_vec_clean[0],
                               target_1_vec_clean[1] + target_2_vec_clean[1],
                               target_1_vec_clean[2] + target_2_vec_clean[2])
    
    has_sequential_move = bool(world_move_2) # move_2 varsa, hareket 50/50 bölünür

    # 4. GÜNCELLENDİ (v11.5): Döngü 0'dan 20'ye (dahil) başlıyor (Toplam 21 frame)
    for i in range(0, 21):
        obj_pos_i = obj_path_21[i] # (obj_x, obj_y, obj_z)
        
        # İlerleme (0.0'dan 1.0'a)
        t_linear = i / num_frames
        
        # Ana easing'i burada hesapla ve tüm enterpolasyonlar için kullan
        t_eased = apply_easing(t_linear, easing)
        
        # --- KAMERA POZİSYON HESAPLAMA PİPELINE'I ---
        
        look_offset_current = (look_offset[0] * t_eased, look_offset[1] * t_eased)
        
        # --- A. Temel Pozisyonu Hesapla (World Move) ---
        # i=0 (t_eased=0) iken cam_pos_base = start_pos (0,0,0)
        cam_pos_base = start_pos
        
        if has_sequential_move:
            if t_linear <= 0.5:
                t_segment_linear = t_linear * 2.0
                t_segment_eased = apply_easing(t_segment_linear, easing) # Easing 1
                cam_pos_base = lerp_3d(start_pos, target_1_vec_clean, t_segment_eased)
            else:
                t_segment_linear = (t_linear - 0.5) * 2.0
                t_segment_eased = apply_easing(t_segment_linear, easing) # Easing 2
                cam_pos_base = lerp_3d(target_1_vec_clean, target_2_combined_clean, t_segment_eased)
        
        else: # Sadece Move 1 (veya hiç hareket yok)
            cam_pos_base = lerp_3d(start_pos, target_1_vec_clean, t_eased)

        # --- B. Son Pozisyonu Hesapla (Local Dolly / Push-In) ---
        
        # --- YENİ v11.5: Look-at hedefi enterpolasyonu ---
        
        # 1. Bitiş hedefi (t=1): Obje + DSL offset
        obj_target_pos_end = (
            obj_pos_i[0] + look_offset_current[0],
            obj_pos_i[1] + look_offset_current[1],
            obj_pos_i[2]
        )
        
        obj_target_pos = obj_target_pos_end # Varsayılan (eğer pan yoksa)

        if has_framing_offset:
            # 2. Başlangıç hedefi (t=0): Bitiş hedefi + Yatay (Pan) Ofseti
            obj_target_pos_start = (
                obj_target_pos_end[0] + framing_offset_x, # Pan ofsetini ekle
                obj_target_pos_end[1],
                obj_target_pos_end[2]
            )
            
            # 3. Hedefi 't_eased' ile start -> end arasında enterpole et (Pan hareketi)
            obj_target_pos = lerp_3d(obj_target_pos_start, obj_target_pos_end, t_eased)
        # --- Bitiş v11.5 ---


        # Bakış vektörünü (normalize edilmiş) al (Temel pozisyondan enterpole edilmiş hedefe)
        look_at_vector = (obj_target_pos[0] - cam_pos_base[0], 
                          obj_target_pos[1] - cam_pos_base[1], 
                          obj_target_pos[2] - cam_pos_base[2])
        
        v_norm, magnitude = normalize_vector(look_at_vector)

        # 'push_amount'u hesapla
        t_eased_push = t_eased # Ana easing'i kullan
        push_dir, push_amount_parsed = local_dolly
        current_push_amount = 0.0
        
        if push_dir == 'in':
            current_push_amount = push_amount_parsed * t_eased_push
        elif push_dir == 'out':
            current_push_amount = -push_amount_parsed * t_eased_push
        
        # 'push'u temel pozisyona ekle (bakış vektörü boyunca)
        cam_pos_pushed = (
            cam_pos_base[0] + v_norm[0] * current_push_amount,
            cam_pos_base[1] + v_norm[1] * current_push_amount,
            cam_pos_base[2] + v_norm[2] * current_push_amount
        )
        
        # --- C. YENİ (v11.5): Pozisyonel Ofsetleri Ekle ---
        cam_pos_offsetted = (
            cam_pos_pushed[0] + framing_offset_x,
            cam_pos_pushed[1] + vertical_offset_y,
            cam_pos_pushed[2]
        )

        # --- D. Jitter (Titreme) Ekle ---
        if jitter_style != 'none':
            j_dx, j_dy, j_dz = get_jitter_offset(jitter_style)
            cam_pos_final = (cam_pos_offsetted[0] + j_dx, 
                             cam_pos_offsetted[1] + j_dy, 
                             cam_pos_offsetted[2] + j_dz)
        else:
            cam_pos_final = cam_pos_offsetted
            
        # --- HESAPLAMA PİPELINE'I BİTTİ ---


        # --- Adım 7: Rotasyonu Hesapla ('Look At' + 'Dutch Angle') ---
        
        # 7a. 'look_at' hesapla (Son pozisyondan (cam_pos_final) enterpole edilmiş hedefe (obj_target_pos))
        # i=0 iken:
        # cam_pos_final = (framing_offset_x, vertical_offset_y, 0)
        # obj_target_pos = (obj_pos_0 + framing_offset_x)
        # Bu, doğru "pan" ve "tilt" rotasyonunu verecektir.
        rw_look, rz_look, rx_look, ry_look = calculate_look_at_quaternion(
            cam_pos_final, 
            obj_target_pos, # Artık v11.5 enterpole edilmiş hedef
            fixed_axis # 'pan_only' / 'tilt_only' / None
        )
        q_look = Rotation.from_quat([rx_look, ry_look, rz_look, rw_look])
        
        q_final = q_look # Başlangıç

        # 7b. 'dutch_angle' uygula (post-multiply)
        if dutch_angle_deg != 0:
            t_eased_roll = t_eased # Ana easing'i kullan
            current_dutch_deg = lerp(0.0, dutch_angle_deg, t_eased_roll)
            
            q_roll = Rotation.from_euler('z', current_dutch_deg, degrees=True)
            q_final = q_look * q_roll
        
        # 7c. Sonucu (w, z, x, -ry) formatına geri çevir
        final_quat_xyzw = q_final.as_quat() # (x, y, z, w)
        rw, rx, ry, rz = final_quat_xyzw[3], final_quat_xyzw[0], final_quat_xyzw[1], final_quat_xyzw[2]
        
        # Adım 8: Frame'i Ekle
        tx_final, ty_final, tz_final = cam_pos_final
        camera_frames.append( (rw, rz, rx, -ry, tx_final, ty_final, tz_final) )
        
    return camera_frames



def orbit_track(
    obj_points_float,
    plane='y',                  # DSL: fixed_axis_x
    degrees=90,                 # DSL: deg_90
    direction='cw',             # DSL: cw
    spiral=(None, 0.0),         # DSL: ('in', 0.2)
    # --- COMMON MODIFIERS ---
    easing='linear',
    dutch_angle_deg=0,
    jitter_style='none',
    # --- YENİ (v11) ---
    vertical_angle='none',       # 'aerial', 'low_angle', veya 'none'
    framing_offset='none'        # GÜNCELLENDİ: Artık 'object_left'/'object_right' olabilir
    ):
    """
    DSL (Bölüm 4) ile uyumlu, gelişmiş 'orbit_track' fonksiyonu.
    GÜNCELLENDİ (v11.1): Frame 0 artık 'identity' değil, 't=0' anındaki
                     ofsetli (vertical_angle) pozisyondur.
    GÜNCELLENDİ (v11.2): 'framing_offset' (object_left/right) desteği eklendi.
                     Bu etiket varsa, kamera pozisyonu yatayda (X) kaydırılır
                     ve rotasyon, objeyi merkezlemek için zamanla 'pan' yapar.
    """
    
    # 1. 21 frame'lik obje yolunu al (0-20 indeksli)
    obj_path_21 = interpolate_object_path(obj_points_float)
    camera_frames = [] # Boş liste ile başla
    
    # 2. Frame 0 (Identity) kaldırıldı.
    
    # 3. Orbit parametreleri
    num_frames = 20.0 # float bölme için (0'dan 20'ye kadar olan adımlar)
    end_angle_rad = np.radians(degrees)
    direction_mult = -1.0 if direction == 'cw' else 1.0
    
    # Yörünge yarıçapını Z_OFFSET olarak belirle
    radius = Z_OFFSET 
    if radius <= 1e-9: radius = 0.5 # 0 olamaz, fallback
    obj_pos_0 = obj_path_21[0] # Objenin başlangıç pozisyonu
    
    # YENİ (v11): Dikey (Y) Ofseti Hesapla (aerial/low_angle)
    vertical_offset_amount = 50.0 / 256.0 # (50 birim / 256 ölçek)
    vertical_offset_y = 0.0
    if vertical_angle == 'aerial':
        vertical_offset_y = vertical_offset_amount
    elif vertical_angle == 'low_angle':
        vertical_offset_y = -vertical_offset_amount
        
    # YENİ (v11.2): Yatay (X) Ofseti Hesapla (object_left/right)
    framing_offset_amount = 50.0 / 256.0 # (50 birim / 256 ölçek)
    framing_offset_x = 0.0
    if framing_offset == 'object_left':
        framing_offset_x = framing_offset_amount # Kamerayı Sağa al (Obje Solda)
    elif framing_offset == 'object_right':
        framing_offset_x = -framing_offset_amount # Kamerayı Sola al (Obje Sağda)
        
    has_framing_offset = (framing_offset_x != 0.0)
    
    
    # 4. GÜNCELLENDİ: Döngü 0'dan 20'ye (dahil) başlıyor (Toplam 21 frame)
    for i in range(0, 21):
        obj_pos_i_actual = obj_path_21[i] # Objenin o anki gerçek konumu
        obj_delta = (obj_pos_i_actual[0] - obj_pos_0[0],
                     obj_pos_i_actual[1] - obj_pos_0[1],
                     obj_pos_i_actual[2] - obj_pos_0[2])
        
        # İlerleme (0.0'dan 1.0'a)
        t_linear = i / num_frames
        
        # Easing'i uygula (tüm hareketler için bu 't' kullanılacak)
        t_eased = apply_easing(t_linear, easing)
        
        # --- KAMERA POZİSYON HESAPLAMA PİPELINE'I ---

        # --- A. Yörünge Pozisyonunu Hesapla (Orbit) ---
        
        current_angle = direction_mult * t_eased * end_angle_rad
        
        tx_orbit, ty_orbit, tz_orbit = 0.0, 0.0, 0.0
        
        # Sizin faz kaydırmalı (phase-shifted) mantığınız:
        if plane == 'x': # X etrafında dön (YZ düzleminde)
            ty_orbit = radius * np.sin(current_angle)
            tz_orbit = (radius * np.cos(current_angle)) - radius 
        elif plane == 'z': # Z etrafında dön (XY düzleminde)
            tx_orbit = radius * np.sin(current_angle) 
            ty_orbit = (-radius * np.cos(current_angle)) + radius 
        else: # Varsayılan: plane == 'y' (XZ düzleminde)
            tx_orbit = radius * np.sin(current_angle)
            tz_orbit = (radius * np.cos(current_angle)) - radius 
        
        cam_pos_orbit = (
            tx_orbit + obj_delta[0] + framing_offset_x, # <-- YENİ (v11.2) X Ofseti
            ty_orbit + obj_delta[1] + vertical_offset_y, # <-- v11 Y Ofseti
            tz_orbit + obj_delta[2]
        )

        # --- B. Spiral (Local Dolly) Pozisyonunu Hesapla ---
        
        # Bakış vektörünü al (Yörünge pozisyonundan HEDEFE -> 'obj_pos_i_actual')
        look_at_vector = (obj_pos_i_actual[0] - cam_pos_orbit[0], 
                          obj_pos_i_actual[1] - cam_pos_orbit[1], 
                          obj_pos_i_actual[2] - cam_pos_orbit[2])
        
        v_norm, magnitude = normalize_vector(look_at_vector)

        # 'spiral' miktarını hesapla
        spiral_dir, spiral_amount_parsed = spiral
        current_spiral_amount = 0.0
        
        if spiral_dir == 'in':
            current_spiral_amount = spiral_amount_parsed * t_eased
        elif spiral_dir == 'out':
            current_spiral_amount = -spiral_amount_parsed * t_eased
        
        # 'spiral'ı yörünge pozisyonuna ekle (bakış vektörü boyunca)
        cam_pos_spiraled = (
            cam_pos_orbit[0] + v_norm[0] * current_spiral_amount,
            cam_pos_orbit[1] + v_norm[1] * current_spiral_amount,
            cam_pos_orbit[2] + v_norm[2] * current_spiral_amount
        )

        # --- C. Jitter (Titreme) Ekle ---
        if jitter_style != 'none':
            j_dx, j_dy, j_dz = get_jitter_offset(jitter_style)
            cam_pos_final = (cam_pos_spiraled[0] + j_dx, 
                             cam_pos_spiraled[1] + j_dy, 
                             cam_pos_spiraled[2] + j_dz)
        else:
            cam_pos_final = cam_pos_spiraled
            
        # --- HESAPLAMA PİPELINE'I BİTTİ ---

        
        # --- Adım 7: Rotasyonu Hesapla ('Look At' + 'Dutch Angle') ---
        
        # YENİ (v11.2): Pan (Yatay Kaydırma) için 'look_at' hedefini ayarla
        
        # Hedef her zaman objenin gerçek konumudur
        obj_target_pos_end = obj_pos_i_actual
        obj_target_pos = obj_target_pos_end # Varsayılan

        if has_framing_offset:
            # t=0'da kamera (X ofsetli) "pan yapmamış" hedefe (X ofsetli) bakmalı
            obj_target_pos_start = (
                obj_pos_i_actual[0] + framing_offset_x, 
                obj_pos_i_actual[1], 
                obj_pos_i_actual[2]
            )
            
            # 'look_at' hedefini t=0 (start) ve t=1 (end) arasında enterpole et
            # t_eased=0 -> obj_target_pos = obj_target_pos_start (Pan yok)
            # t_eased=1 -> obj_target_pos = obj_target_pos_end (Tam pan)
            obj_target_pos = lerp_3d(obj_target_pos_start, obj_target_pos_end, t_eased)
        
        
        # 7a. 'look_at' hesapla (Son pozisyondan hedefe)
        # Hedef (obj_target_pos) artık enterpole edildiği için,
        # 'calculate_look_at' otomatik olarak "pan" hareketini zamanla yapar.
        rw_look, rz_look, rx_look, ry_look = calculate_look_at_quaternion(
            cam_pos_final, 
            obj_target_pos, # Enterpole edilmiş hedef
            fixed_axis=None # Orbit'te 'look_at' her zaman serbesttir
        )
        q_look = Rotation.from_quat([rx_look, ry_look, rz_look, rw_look])
        
        q_final = q_look # Başlangıç

        # 7b. 'dutch_angle' uygula (post-multiply)
        if dutch_angle_deg != 0:
            # Easing'i Dutch Angle'a da uygula
            current_dutch_deg = lerp(0.0, dutch_angle_deg, t_eased)
            
            q_roll = Rotation.from_euler('z', current_dutch_deg, degrees=True)
            q_final = q_look * q_roll
        
        # 7c. Sonucu (w, z, x, -ry) formatına geri çevir
        final_quat_xyzw = q_final.as_quat() # (x, y, z, w)
        rw, rx, ry, rz = final_quat_xyzw[3], final_quat_xyzw[0], final_quat_xyzw[1], final_quat_xyzw[2]
        
        # Adım 8: Frame'i Ekle
        tx_final, ty_final, tz_final = cam_pos_final
        camera_frames.append( (rw, rz, rx, -ry, tx_final, ty_final, tz_final) )
        
    return camera_frames

# --- `main` Fonksiyonunuzun İçindeki Güncellenmiş Çağrı Bloğu ---
# (main_parser_example fonksiyonu silindi ve mantığı main'e taşındı)

def main():
    # 1. Argümanları ayarla (Sizin sağladığınız kısım)
    parser = argparse.ArgumentParser(
        description="Obje yörüngelerini normalize eder, kamera yörüngeleri üretir "
                    "ve sonucu [0, 512] uzayına geri dönüştürür."
    )
    parser.add_argument(
        "-obj", "--object_trajectories",
        default="generated_bbox_trajectories_test.txt",
        help="Girdi obje yörünge dosyası (örn: generated_bbox_trajectories_v6.txt)"
    )
    parser.add_argument(
        "-tag", "--camera_tags",
        default="generated_relative_tags_test.txt",
        help="Girdi kamera etiket dosyası (örn: generated_relative_camera_tags.txt)"
    )
    parser.add_argument(
        "-o", "--output",
        default="generated_relative_trajectories_test.txt",
        help="Çıktı kamera yörünge dosyası (örn: generated_camera_trajectories_v6.txt)"
    )
    args = parser.parse_args()

    print("Yörüngeler üretiliyor (DSL v11 Parser)...") # GÜNCELLENDİ
    line_count = 0
    
    try:
        # 2. Dosyaları aç
        with open(args.object_trajectories, 'r', encoding='utf-8') as obj_file, \
             open(args.camera_tags, 'r', encoding='utf-8') as tag_file, \
             open(args.output, 'w', encoding='utf-8') as out_file:
            
            for obj_line, tag_line in zip(obj_file, tag_file):
                line_count += 1
                
                # Adım 3: Obje yörüngesini [0, 512] tamsayı olarak oku
                obj_points_int = parse_object_trajectory(obj_line)
                tag = tag_line.strip()
                
                if obj_points_int is None:
                    print(f"Uyarı: {line_count}. satırdaki obje yörüngesi atlanıyor (parse hatası).")
                    continue
                
                # Adım 4: Objeyi [-1, 1] float uzayına işle
                obj_points_float = process_object_trajectory(obj_points_int)
                
                # Adım 5: (GÜNCELLENMİŞ v11 DSL PARSER) Etiketi işle
                camera_frames_float = []
                
                # Parametreleri tag'den ayrıştır
                tag_parts = tag.split()
                base_behavior = tag_parts[0] if tag_parts else ""

                # --- Common Modifiers (Tüm tracking'ler için) ---
                # Önce Common Modifiers'ı ayrıştır
                easing = 'linear'
                dutch_angle_deg = 0
                jitter_style = 'none'
                vertical_angle = 'none'   # YENİ (v11): aerial, low_angle
                framing_offset = 'none'   # YENİ (v11): object_left, object_right

                for part in tag_parts:
                    if part.startswith('ease_'):
                        easing = part
                    elif part.startswith('dutch_'):
                        try:
                            parts = part.split('_')
                            if len(parts) == 3:
                                direction = parts[1]
                                angle_magnitude = int(parts[2])
                                if direction == 'ccw':
                                    dutch_angle_deg = -angle_magnitude
                                elif direction == 'cw':
                                    dutch_angle_deg = angle_magnitude
                                else:
                                    print(f"Uyarı: Geçersiz dutch yönü '{direction}' şurada: '{part}'")
                                    continue
                            else:
                                print(f"Uyarı: Beklenmeyen dutch formatı '{part}'.")
                                continue
                        except ValueError:
                            print(f"Hata: 'dutch_' etiketinde açıya dönüştürülemeyen değer: '{part}'")
                            continue
                        except Exception: pass
                    elif part.startswith('jitter_'):
                        jitter_style = part.split('_')[1] # 'low' veya 'high'
                
                    # YENİ (v11) Common Modifiers
                    elif part == 'aerial' or part == 'low_angle':
                        vertical_angle = part
                    elif part == 'object_left' or part == 'object_right':
                        framing_offset = part

                
                # --- Base Behavior'a Göre Ayrıştırma ---

                if base_behavior == "tail_track":
                    
                    # 1. DSL (v11) için varsayılan değerleri ayarla
                    follow_style = 'hard'
                    follow_axis = [] 
                    look_at = True  # GÜNCELLENDİ (v11): Default True
                    dolly = (None, 0.0)
                    amplitude = [1.0, 1.0, 1.0]
                    mirror_axis = []
                    is_lead = False # YENİ (v11)
                    
                    # 2. 'tag_parts' (etiket listesi) içindeki her etiketi ayrıştır
                    for part in tag_parts:
                        
                        # --- Tail Modifiers ---
                        if part.startswith("follow_style_"):
                            follow_style = part.split('_')[-1] # 'hard', 'soft', 'lazy'
                        
                        elif part.startswith("follow_axis_"):
                            follow_axis.append(part.split('_')[-1]) # 'x', 'y', 'z'
                        
                        # GÜNCELLENDİ (v11)
                        elif part == "dont_look":
                            look_at = False
                        
                        # YENİ (v11)
                        elif part == "lead":
                            is_lead = True
                            
                        elif part.startswith("dolly_in_") or part.startswith("dolly_out_"):
                            try:
                                direction = 'in' if part.startswith("dolly_in_") else 'out'
                                amount = float(part.split('_')[-1])
                                dolly = (direction, amount)
                            except Exception: pass
                        
                        elif part.startswith("amp_x_"):
                            try: amplitude[0] = float(part.split('_')[-1])
                            except Exception: pass
                        elif part.startswith("amp_y_"):
                            try: amplitude[1] = float(part.split('_')[-1])
                            except Exception: pass
                        elif part.startswith("amp_z_"):
                            try: amplitude[2] = float(part.split('_')[-1])
                            except Exception: pass
                        elif part.startswith("amp_all_"):
                            try: 
                                val = float(part.split('_')[-1])
                                amplitude = [val, val, val]
                            except Exception: pass
                        
                        elif part.startswith("mirror_"):
                            mirror_axis.append(part.split('_')[-1]) # 'x', 'y'

                    # 3. Güncellenmiş 'tail_track' fonksiyonunu YENİ parametrelerle çağır
                    # (Fonksiyonun bu parametreleri kabul ettiği varsayılıyor)
                    camera_frames_float = tail_track(
                        obj_points_float, 
                        follow_style=follow_style,
                        follow_axis=follow_axis,
                        look_at=look_at,
                        dolly=dolly,
                        amplitude=tuple(amplitude),
                        mirror_axis=mirror_axis,
                        # Common Modifiers
                        easing=easing,
                        dutch_angle_deg=dutch_angle_deg,
                        jitter_style=jitter_style,
                        lead=is_lead,
                        vertical_angle=vertical_angle,
                        framing_offset=framing_offset
                    )
                
                elif base_behavior == "rotation_track":
                    
                    # 1. DSL (Bölüm 6) için varsayılan değerleri ayarla
                    rotation_axis = 'full'
                    local_dolly = (None, 0.0)
                    world_move_1 = {} # {'x': 0.2, 'y': -0.1}
                    world_move_2 = {}
                    look_offset = [0.0, 0.0] # [x, y]

                    # 2. 'tag_parts' içindeki her etiketi ayrıştır
                    for part in tag_parts:
                        
                        if part == 'pan_only' or part == 'tilt_only':
                            rotation_axis = part
                        
                        elif part.startswith('push_in_') or part.startswith('push_out_'):
                            try:
                                direction = 'in' if part.startswith("push_in_") else 'out'
                                amount = float(part.split('_')[-1])
                                local_dolly = (direction, amount)
                            except Exception: pass
                        
                        elif part.startswith('look_offset_x_'):
                            try: look_offset[0] = float(part.split('_')[-1])
                            except Exception: pass
                        elif part.startswith('look_offset_y_'):
                            try: look_offset[1] = float(part.split('_')[-1])
                            except Exception: pass
                        
                        # World Moves (Regex en güvenlisi)
                        elif re.match(r"(truck|pedestal|goes)_(right|left|up|down|in|out)_([12])_(\d\.\d)", part):
                            try:
                                parts_match = part.split('_')
                                move_type, direction, set_num, amount = parts_match[0], parts_match[1], parts_match[2], float(parts_match[3])
                                
                                axis = 'x' if direction in ['right', 'left'] else ('y' if direction in ['up', 'down'] else 'z')
                                amount = -amount if direction in ['left', 'down', 'out'] else amount
                                
                                if set_num == '1':
                                    world_move_1[axis] = amount
                                else:
                                    world_move_2[axis] = amount
                            except Exception: pass

                    # 3. Yeni 'rotation_track' fonksiyonunu çağır
                    # (Fonksiyonun bu parametreleri kabul ettiği varsayılıyor)
                    camera_frames_float = rotation_track(
                        obj_points_float,
                        rotation_axis=rotation_axis,
                        local_dolly=local_dolly,
                        world_move_1=world_move_1,
                        world_move_2=world_move_2,
                        look_offset=tuple(look_offset),
                        # Common Modifiers
                        easing=easing,
                        dutch_angle_deg=dutch_angle_deg,
                        jitter_style=jitter_style,
                        vertical_angle=vertical_angle,
                        framing_offset=framing_offset
                    )
                
                elif base_behavior == "orbit_track":
                    
                    # 1. DSL (Bölüm 4) için varsayılan değerleri ayarla
                    plane = 'y' # 'fixed_axis_y' varsayılan
                    degrees = 90
                    direction = 'ccw'
                    spiral = (None, 0.0)

                    # 2. 'tag_parts' içindeki her etiketi ayrıştır
                    for part in tag_parts:
                        
                        if part.startswith('plane_axis_'):
                            plane = part.split('_')[-1] # x, y, z
                        
                        elif part.startswith('deg_'):
                            try: degrees = int(part.split('_')[-1])
                            except Exception: pass
                        
                        elif part == 'cw' or part == 'ccw':
                            direction = part
                        
                        elif part.startswith('spiral_in_') or part.startswith('spiral_out_'):
                            try:
                                direction_spiral = 'in' if part.startswith("spiral_in_") else 'out'
                                amount = float(part.split('_')[-1])
                                spiral = (direction_spiral, amount)
                            except Exception: pass
                    
                    # 3. Yeni 'orbit_track' fonksiyonunu çağır
                    # (Fonksiyonun bu parametreleri kabul ettiği varsayılıyor)
                    camera_frames_float = orbit_track(
                        obj_points_float,
                        plane=plane,
                        degrees=degrees,
                        direction=direction,
                        spiral=spiral,
                        # Common Modifiers
                        easing=easing,
                        dutch_angle_deg=dutch_angle_deg,
                        jitter_style=jitter_style,
                        vertical_angle=vertical_angle,
                        framing_offset=framing_offset
                    )

                elif base_behavior == "free-form":
                    print(f"Uyarı: {line_count}. satırdaki 'free_form' etiketi atlanıyor.")
                    continue
                    
                else:
                    print(f"Uyarı: {line_count}. satırdaki etiket '{tag}' tanınmıyor. Varsayılan (sabit) yörünge kullanılıyor.")
                    camera_frames_float = handle_unknown_tag(obj_points_float)

                # Adım 6: Çıktıyı [0, 512] tamsayı uzayına geri dönüştür
                frame_strings = []
                for frame in camera_frames_float:
                    denorm_frame_values = [denormalize_value(v) for v in frame]
                    frame_strings.append(" ".join(map(str, denorm_frame_values)))
                
                # Adım 7: Dosyaya yaz
                output_line = " | ".join(frame_strings)
                out_file.write(output_line + "\n")


    except FileNotFoundError as e:
        print(f"HATA: Dosya bulunamadı: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"HATA: Beklenmedik bir hata oluştu (satır {line_count}): {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nBaşarıyla tamamlandı. {line_count} kamera yörüngesi üretildi.")
    print(f"Çıktı dosyası: {args.output}")

if __name__ == "__main__":
    main()