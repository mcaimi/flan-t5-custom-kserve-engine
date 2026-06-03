from unittest.mock import patch, MagicMock

from torch import float16, float32


@patch("libs.utils.tc")
def test_get_accelerator_device_cpu(mock_tc):
    mock_tc.is_available.return_value = False
    from libs.utils import get_accelerator_device

    with patch("libs.utils.platform.system", return_value="Linux"):
        accelerator, dtype = get_accelerator_device()

    assert accelerator == "cpu"
    assert dtype == float32


@patch("libs.utils.tc")
def test_get_accelerator_device_cuda(mock_tc):
    mock_tc.is_available.return_value = True
    mock_tc.get_device_name.return_value = "NVIDIA A100"
    mock_tc.get_device_capability.return_value = (8, 0)
    mock_tc.mem_get_info.return_value = (40 * 1024**3, 80 * 1024**3)

    from libs.utils import get_accelerator_device
    accelerator, dtype = get_accelerator_device()

    assert accelerator == "cuda"
    assert dtype == float16


@patch("libs.utils.tc")
def test_get_accelerator_device_mps(mock_tc):
    mock_tc.is_available.return_value = False

    mock_tmps = MagicMock()
    mock_tmps.is_available.return_value = True
    mock_tmps.get_name.return_value = "Apple M1"
    mock_tmps.get_core_count.return_value = 8

    with patch("libs.utils.platform.system", return_value="Darwin"), \
         patch("libs.utils.tmps", mock_tmps):
        from libs.utils import get_accelerator_device
        accelerator, dtype = get_accelerator_device()

    assert accelerator == "mps"
    assert dtype == float16


@patch("libs.utils.tc")
def test_cuda_takes_precedence_over_mps(mock_tc):
    mock_tc.is_available.return_value = True
    mock_tc.get_device_name.return_value = "NVIDIA A100"
    mock_tc.get_device_capability.return_value = (8, 0)
    mock_tc.mem_get_info.return_value = (40 * 1024**3, 80 * 1024**3)

    with patch("libs.utils.platform.system", return_value="Darwin"):
        from libs.utils import get_accelerator_device
        accelerator, dtype = get_accelerator_device()

    assert accelerator == "cuda"
    assert dtype == float16
