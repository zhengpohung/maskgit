# communication_sim.py 的最終版本 (分析模型)
import numpy as np

def calculate_analytical_per(snr_db, L_total_bits=160, code_rate=0.5, M=16):
    """
    根據 ARQ 論文的分析模型計算 PER (修正版)。
    全程使用 dB 單位進行計算以匹配論文的擬合模型。
    """
    # 根據 ARQ 論文 Table I，Code 1, 16QAM 的參數
    k_M = 0.523
    b_M = -0.341

    # 1. 將符號 SNR (Es/N0) 轉換為 位元 Eb/N0 (γ_b)，全程在 dB 域操作
    # SNR_linear = γ_b_linear * code_rate * log2(M)
    # 10*log10(SNR_linear) = 10*log10(γ_b_linear) + 10*log10(code_rate * log2(M))
    # SNR_dB = γ_b_dB + 10*log10(code_rate * log2(M))
    conversion_factor_db = 10 * np.log10(code_rate * np.log2(M))
    conversion_factor_db = 0.88
    gamma_b_db = snr_db - conversion_factor_db

    # 2. 計算門檻值 γ_ω (dB)。論文中的 log 是自然對數 (ln)。
    # L_total_bits 是線性值，log(L_total_bits) 是線性值
    gamma_omega_db = k_M * np.log(L_total_bits) + b_M

    # 3. 計算 PER 的核心公式
    # PER ≈ 1 - exp(-γ_ω / γ_b)
    # 注意，此處的 γ_ω 和 γ_b 是線性值
    gamma_omega_linear = 10**(gamma_omega_db / 10.0)
    gamma_b_linear = 10**(gamma_b_db / 10.0)
    
    per = 1 - np.exp(-gamma_omega_linear / gamma_b_linear)
    
    return np.clip(per, 0, 1)

def simulate_transmission_analytical(token_indices, snr_db, generator):
    """
    接收 token 序列，使用分析模型計算 PER，並隨機丟棄封包來模擬傳輸。
    返回一個可能帶有 MASK 的 token 序列。
    """
   
    # 1. 計算該 SNR 下的理論 PER
    # 每個封包有 16 token，每個 token 10 bits，共 160 bits
    bits_per_packet = 16 * 10
    per_map = {
        6: 0.69,
        8: 0.52,
        10: 0.37,
        12: 0.25,
        14: 0.17,
        16: 0.11,
        18: 0.07
    }

    # 我們使用固定的 seed 來確保發送端和接收端的打亂/還原順序永遠一致。
    _permutation_seed = 42
    _rng = np.random.default_rng(_permutation_seed)
    _total_tokens = 256
    
    # 產生一個從 0 到 255 的隨機排列
    _permutation_indices = _rng.permutation(_total_tokens)

    # 為了能夠還原，我們需要計算出反向的排列順序
    _inverse_permutation_indices = np.empty_like(_permutation_indices)
    _inverse_permutation_indices[_permutation_indices] = np.arange(_total_tokens)
    
    # 從字典中查找 PER。如果輸入的 snr_db 不在字典中，則拋出錯誤。
    per = per_map.get(snr_db)
    if per is None:
        raise ValueError(f"提供的 SNR 值 {snr_db} 不在預設的對應表中 {list(per_map.keys())}。請提供有效的 SNR。")
    # 2. 將 tokens 分割成封包 (邏輯上的)
    # token_indices 應為 (1, 256) 或 (256,)
    token_indices = np.asarray(token_indices).flatten()
    num_packets = 16

    permuted_tokens = token_indices[_permutation_indices]
    tokens_as_packets = np.reshape(permuted_tokens, (num_packets, 16))

    # 3. 根據 PER 決定哪些封包遺失
    lost_packets_mask = np.random.rand(num_packets) < per
    num_packet_errors = np.sum(lost_packets_mask)
    #print(f"SNR={snr_db}dB, PER={per:.4f}, Packet Errors: {num_packet_errors} / {num_packets}")

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
    received_permuted_tokens = received_packets.flatten()
    final_tokens_in_original_order = received_permuted_tokens[_inverse_permutation_indices]
    model_mask_id = generator.maskgit_cf.transformer.mask_token_id
    final_tokens_in_original_order[final_tokens_in_original_order == MASK_TOKEN_ID] = model_mask_id
    
    # 調整資料形狀：從 (256,) 變成 (1, 256) 以符合模型輸入
    input_for_repair = final_tokens_in_original_order.reshape(1, -1)
    
    return input_for_repair
