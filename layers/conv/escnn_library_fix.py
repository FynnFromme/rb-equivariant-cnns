
from escnn.nn.modules.conv import _RdConv

def fix_conv_eval():
    # Detaches the expanded parameters during eval so that the module in eval can be used as a submodule
    # of another module being trained
    def fixed_train(self, mode=True):
            r"""
            
            If ``mode=True``, the method sets the module in training mode and discards the :attr:`~escnn.nn._RdConv.filter`
            and :attr:`~escnn.nn._RdConv.expanded_bias` attributes.
            
            If ``mode=False``, it sets the module in evaluation mode. Moreover, the method builds the filter and the bias
            using the current values of the trainable parameters and store them in :attr:`~escnn.nn._RdConv.filter` and
            :attr:`~escnn.nn._RdConv.expanded_bias` such that they are not recomputed at each forward pass.
            
            .. warning ::
                
                This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
                a mode and lately loading it in a model with a different mode, as the attributes of this class change.
                To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
                dictionary.
            
            Args:
                mode (bool, optional): whether to set training mode (``True``) or evaluation mode (``False``).
                                    Default: ``True``.

            """

            if mode:
                # TODO thoroughly check this is not causing problems
                if hasattr(self, "filter"):
                    del self.filter
                if hasattr(self, "expanded_bias"):
                    del self.expanded_bias
            elif self.training:
                # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
        
                _filter, _bias = self.expand_parameters()
        
                self.register_buffer("filter", _filter.detach())
                if _bias is not None:
                    self.register_buffer("expanded_bias", _bias.detach())
                else:
                    self.expanded_bias = None

            return super(_RdConv, self).train(mode)
    _RdConv.train = fixed_train