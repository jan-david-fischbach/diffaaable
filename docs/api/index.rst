API Reference
===================================

All functions are importable directly from ``diffaaable`` and return the
nodes ``z_j``, values ``f_j``, weights ``w_j`` and poles ``z_n`` of the
barycentric rational approximation.
The approximation can be evaluated with :func:`~diffaaable.tensor.tensor_baryrat`.

=========================================================  ==============================================================
Function                                                   Use it for
=========================================================  ==============================================================
:func:`~diffaaable.core.aaa`                               scalar functions :math:`f(z)`
:func:`~diffaaable.tensor.tensor_aaa`                      tensor-valued functions :math:`\mathbf{F}(z)` with common poles
:func:`~diffaaable.set_aaa.set_aaa`                        vector-valued functions, avoiding repeated large SVDs
:func:`~diffaaable.adaptive.adaptive_aaa`                  expensive functions: locate poles with few evaluations
:func:`~diffaaable.selective.selective_subdivision_aaa`    many poles in a large domain
:func:`~diffaaable.lorentz.lorentz_aaa`                    poles symmetric about the imaginary axis
:func:`~diffaaable.vectorial.vectorial_aaa`                deprecated, use :func:`~diffaaable.tensor.tensor_aaa`
=========================================================  ==============================================================
