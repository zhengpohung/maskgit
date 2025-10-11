# communication_sim.py 的內容範例
import numpy as np
import commpy
import commpy.channelcoding.convcode as cc
import commpy.modulation as modulation
from commpy.channels import awgn
# ... (這裡放入所有 commpy 相關的設定和輔助函式) ...

def simulate_transmission(token_indices, snr_db, generator):
    """
    接收一個 token 序列，模擬整個通訊過程，
    最後返回一個可能帶有 MASK 的 token 序列。
    """
    # 1. 發射端 (TX)
    token_indices = np.asarray(token_indices)
    #    - Token to Bits
    num_bits_per_token = 10
    
    # 將每個 token 索引轉換為 10-bit 的二進位序列
    # 例如: 123 -> [0, 0, 1, 1, 1, 1, 0, 1, 1]
    #message_bits = np.unpackbits(token_indices.astype(np.uint16).byteswap().view(np.uint8))
    # 需要一些 reshaping 和 slicing 來確保每個 token 準確對應 10 bits
    # 這裡只提供概念，實際操作需要仔細處理 NumPy 的位元操作
    # 簡化版概念：
    def token_to_bits(token, num_bits):
        # 將數字轉為二進位字串，並補零
        return [int(b) for b in bin(token)[2:].zfill(num_bits)]
    
    all_bits = np.array([token_to_bits(t, 10) for t in token_indices.flatten()]).flatten()


    #    - Packetization
    bits_per_packet = 16 * 10 # 160 bits per packet
    packets = np.reshape(all_bits, (-1, bits_per_packet))


    #    - Convolutional Coding
    # 定義卷積碼的生成多項式 (這是一個標準範例，可能需要根據論文調整)
    generator_polynomials = np.array([[0o7, 0o5]]) # G(D) = [1+D+D^2, 1+D^2]
    trellis = cc.Trellis(np.array([2]), generator_polynomials)

    coded_packets = []
    for packet in packets:
        coded_bits = cc.conv_encode(packet, trellis)
        coded_packets.append(coded_bits)
    #    - Modulation
    modem = modulation.QAMModem(16)
    modulated_packets = []
    for coded_packet in coded_packets:
        modulated_symbols = modem.modulate(coded_packet)
        modulated_packets.append(modulated_symbols)
    #    ...

    # 2. 通道 (Channel)
    #    - Rayleigh Fading

    received_packets = []
    avg_symbol_energy = modem.Es
    signal_power = avg_symbol_energy
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_variance_per_component = noise_power / 2.0
    for modulated_packet in modulated_packets:
        # 1. 瑞利衰減通道 (Rayleigh Fading Channel)
        # 產生一個複數高斯隨機變數，其振幅服從瑞利分佈
        rayleigh_coeff = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        faded_signal = modulated_packet * rayleigh_coeff

        # 2. 添加高斯白雜訊 (AWGN)
        # commpy.utilities.awgn 函式可以幫我們完成
        
        noise = (np.sqrt(noise_variance_per_component) * np.random.randn(*faded_signal.shape) + 
                 1j * np.sqrt(noise_variance_per_component) * np.random.randn(*faded_signal.shape))
        
        noisy_signal = faded_signal + noise
        
        # 接收端需要知道衰減係數來進行等化
        # 在實際系統中這一步是通道估計，模擬時我們可以假設接收端完美知道
        equalized_signal = noisy_signal / rayleigh_coeff
        equalized_noise_power = noise_power / (np.abs(rayleigh_coeff)**2)
        received_packets.append({
            "signal": equalized_signal,
            "noise_power": equalized_noise_power
        })
    #    - AWGN
    #    ...

    # 3. 接收端 (RX)
    #    - Demodulation (soft)
    demod_llrs = []
    # [修正] 使用列表推導式
    for rp_data in received_packets:
        # 將每個封包 "自己對應的" 雜訊功率傳入解調器
        llrs = modem.demodulate(rp_data["signal"], demod_type='soft', noise_var=rp_data["noise_power"])
        demod_llrs.append(llrs)
    
    demod_llrs = np.array(demod_llrs)

    # [修正] Viterbi 解碼後，必須將輸出裁切回原始封包長度
    decoded_packets = np.array([cc.viterbi_decode(llrs, trellis, decoding_type='soft')[:bits_per_packet] for llrs in demod_llrs])


    #    - CRC Check & Depacketization
        # 假設 MASK_TOKEN_ID 是一個預留的特殊 ID，例如 1024
    MASK_TOKEN_ID = -1
    received_token_indices = []

    num_packet_errors = 0
    for i, decoded_packet in enumerate(decoded_packets):
        original_packet_bits = packets[i]
        if not np.array_equal(decoded_packet, original_packet_bits):
            # 封包出錯，生成 16 個 MASK token
            num_packet_errors += 1
            received_token_indices.extend([MASK_TOKEN_ID] * 16)
        else:
            # 同樣，這裡的位元到位元組轉換需要精確處理
            # 簡化版概念：
            for j in range(16):
                token_bits = decoded_packet[j*10 : (j+1)*10]
                token_val = int("".join(map(str, token_bits)), 2)
                received_token_indices.append(token_val)

    print(f"Total packet errors: {num_packet_errors} / {len(packets)}")

    # 將 received_token_indices 轉為 NumPy 陣列，準備送入 MaskGIT
    final_tokens = np.array(received_token_indices)


    # --- 4. 格式化輸出 (將 prepare_masked_input 的邏輯整合進來) ---
    
    # 從 generator 物件獲取模型官方的 MASK ID
    model_mask_id = getattr(generator, 'mask_token_id', MASK_TOKEN_ID)

    # 替換 MASK ID
    final_tokens[final_tokens == MASK_TOKEN_ID] = model_mask_id
    
    # 調整資料形狀：從 (256,) 變成 (1, 256)
    input_for_repair = final_tokens.reshape(1, -1)
    
    return input_for_repair
