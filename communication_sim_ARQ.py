# communication_sim.py 的最終版本 (分析模型)
import numpy as np

def calculate_analytical_per(snr_db, L_total_bits=160, code_rate=0.5, M=16):
    """
    [cite_start]根據 ARQ 論文 [cite: 1] 的分析模型計算 PER。
    論文參考: Energy_Efficiency_and_Spectral_Efficiency_Tradeoff_in_Type-I_ARQ_Systems.pdf
    """
    # [cite_start]根據 ARQ 論文 Table I，Code 1, 16QAM 的參數 [cite: 166]
    k_M = 0.523
    b_M = -0.314

    # SNR 是 Es/N0，模型需要 Eb/N0 (即 γ_b)。進行轉換。
    # Es/N0 = (Eb/N0) * code_rate * log2(M)
    snr_linear = 10**(snr_db / 10.0)
    gamma_b_linear = snr_linear / (code_rate * np.log2(M))

    # [cite_start]計算門檻值 γ_ω，論文中 log 為自然對數 [cite: 157]
    gamma_omega = k_M * np.log(L_total_bits) + b_M

    # [cite_start]計算 PER 的核心公式 [cite: 151]
    per = 1 - np.exp(-gamma_omega / gamma_b_linear)
    
    return np.clip(per, 0, 1)

def simulate_transmission_analytical(token_indices, snr_db, generator):
    """
    接收 token 序列，使用分析模型計算 PER，並隨機丟棄封包來模擬傳輸。
    返回一個可能帶有 MASK 的 token 序列。
    """
    # 1. 計算該 SNR 下的理論 PER
    # 每個封包有 16 token，每個 token 10 bits，共 160 bits
    bits_per_packet = 16 * 10
    per = calculate_analytical_per(snr_db, L_total_bits=bits_per_packet)
    
    # 2. 將 tokens 分割成封包 (邏輯上的)
    # token_indices 應為 (1, 256) 或 (256,)
    token_indices = np.asarray(token_indices).flatten()
    num_packets = 16
    tokens_as_packets = np.reshape(token_indices, (num_packets, 16)) # 16x16

    # 3. 根據 PER 決定哪些封包遺失
    lost_packets_mask = np.random.rand(num_packets) < per
    num_packet_errors = np.sum(lost_packets_mask)
    print(f"SNR={snr_db}dB, PER={per:.4f}, Packet Errors: {num_packet_errors} / {num_packets}")

    # 4. 建立接收端的 token 序列
    MASK_TOKEN_ID = -1 # 使用一個臨時 ID
    
    # 根據丟包遮罩，將遺失的封包替換為 MASK_TOKEN_ID
    # 使用 np.where，如果 lost_packets_mask[i] 是 True，就用 MASK 填充，否則用原始封包
    received_packets = np.where(
        lost_packets_mask[:, np.newaxis], # 擴展維度以匹配 tokens_as_packets
        np.full_like(tokens_as_packets, MASK_TOKEN_ID),
        tokens_as_packets
    )

    # 5. 將封包重新組合回 token 序列並格式化輸出
    final_tokens = received_packets.flatten()
    
    model_mask_id = generator.maskgit_cf.transformer.mask_token_id
    final_tokens[final_tokens == MASK_TOKEN_ID] = model_mask_id
    
    # 調整資料形狀：從 (256,) 變成 (1, 256) 以符合模型輸入
    input_for_repair = final_tokens.reshape(1, -1)
    
    return input_for_repair