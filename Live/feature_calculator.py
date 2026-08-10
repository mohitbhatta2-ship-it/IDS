import numpy as np


def safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def safe_std(values):
    return float(np.std(values)) if len(values) > 1 else 0.0


def safe_var(values):
    return float(np.var(values)) if len(values) > 1 else 0.0


def safe_max(values):
    return float(max(values)) if values else 0.0


def safe_min(values):
    return float(min(values)) if values else 0.0


def iat_stats(times):
    """
    Returns:
        total, mean, std, min, max
    """

    if len(times) < 2:
        return 0, 0, 0, 0, 0

    iats = np.diff(times) * 1_000_000

    return (
        float(np.sum(iats)),
        float(np.mean(iats)),
        float(np.std(iats)),
        float(np.min(iats)),
        float(np.max(iats))
    )


def calculate_features(flow):
    # Two feature definitions are less obvious than their names suggest, and both
    # were wrong before -- matching CICFlowMeter here matters more than the name:
    #
    #   * "Fwd Seg Size Min" is the minimum forward *header* length (min_seg_size_
    #     forward), not a payload size. Its training values are 8/20/32/40 bytes --
    #     UDP and TCP header sizes -- so it is read from forward_header_lengths.
    #   * "Init Fwd Win Byts" is -1, not 0, when the flow has no forward TCP
    #     window (UDP, or a flow captured mid-stream); the training data uses -1.

    duration_sec = flow.last_seen - flow.start_time

    if duration_sec <= 0:
        duration_sec = 1e-6

    duration = duration_sec * 1_000_000

    flow_iat_total, flow_iat_mean, flow_iat_std, flow_iat_min, flow_iat_max = \
        iat_stats(flow.all_packet_times)

    fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_min, fwd_iat_max = \
        iat_stats(flow.forward_packet_times)

    features = {

        "Init Fwd Win Byts":
            flow.init_fwd_win_bytes if flow.init_fwd_win_bytes is not None else -1,

        "Fwd IAT Tot":
            fwd_iat_total,

        "Fwd IAT Max":
            fwd_iat_max,

        "Dst Port":
            flow.dst_port,

        "Fwd IAT Mean":
            fwd_iat_mean,

        "Fwd Header Len":
            sum(flow.forward_header_lengths),

        "Flow IAT Max":
            flow_iat_max,

        "Flow Duration":
            duration,

        "Fwd Pkts/s":
            flow.forward_packets / duration_sec,

        "TotLen Fwd Pkts":
            flow.forward_bytes,

        "Subflow Fwd Byts":
            flow.forward_bytes,

        "Fwd Pkt Len Mean":
            safe_mean(flow.forward_packet_lengths),

        "Fwd Seg Size Avg":
            safe_mean(flow.forward_packet_lengths),

        "Fwd Pkt Len Max":
            safe_max(flow.forward_packet_lengths),

        "Flow IAT Mean":
            flow_iat_mean,

        "Flow Pkts/s":
            flow.total_packets / duration_sec,

        "Pkt Len Max":
            safe_max(flow.all_packet_lengths),

        "Pkt Len Mean":
            safe_mean(flow.all_packet_lengths),

        "Fwd IAT Min":
            fwd_iat_min,

        "Bwd Pkt Len Mean":
            safe_mean(flow.backward_packet_lengths),

        "Pkt Size Avg":
            safe_mean(flow.all_packet_lengths),

        "Bwd Seg Size Avg":
            safe_mean(flow.backward_packet_lengths),

        "Bwd Pkt Len Max":
            safe_max(flow.backward_packet_lengths),

        "TotLen Bwd Pkts":
            flow.backward_bytes,

        "Subflow Bwd Byts":
            flow.backward_bytes,

        "Pkt Len Var":
            safe_var(flow.all_packet_lengths),

        "Pkt Len Std":
            safe_std(flow.all_packet_lengths),

        "Flow IAT Min":
            flow_iat_min,

        "Fwd Seg Size Min":
            safe_min(flow.forward_header_lengths),

        "Fwd Pkt Len Std":
            safe_std(flow.forward_packet_lengths)
    }

    return features