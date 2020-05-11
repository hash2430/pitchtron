from torch import nn
import torch
import torch.nn.functional as F

class AngularMarginSoftmaxLoss(nn.Module):
    def __init__(self, speaker_embed_dim, n_speakers, margin, scaler, eps=0.00001):
        super(AngularMarginSoftmaxLoss, self).__init__()
        self.W = nn.Parameter(torch.Tensor(n_speakers, speaker_embed_dim))# []
        nn.init.xavier_uniform(self.W)
        self.margin = margin
        self.scaler = scaler
        self.speaker_embed_dim = speaker_embed_dim
        self.n_speakers = n_speakers
        self.eps = eps
    def forward(self, _speaker_embedding, _true_label_index):
        loss = 0
        self.W = torch.nn.Parameter(F.normalize(self.W, dim=1))
        # Average pool speaker embedding on time domain
        # Here, speaker classifier input gst has no time dim so speaker embedding doesnt have time dim either.
        # _speaker_embedding = _speaker_embedding.transpose(1, 2)
        # _speaker_embedding = F.avg_pool1d(_speaker_embedding, kernel_size=_speaker_embedding.shape[-1]).squeeze()
        # _speaker_embedding = F.normalize(_speaker_embedding, dim=1)


        true_label_index = _true_label_index
        speaker_embedding = _speaker_embedding.squeeze()
        # numerator = torch.matmul(self.W,speaker_embedding)[true_label_index].squeeze() - self.margin #[6,scalar]
        numerator = torch.matmul(speaker_embedding, self.W.transpose(0,1))
        numerator = [numerator[i][true_label_index[i]] for i in range(len(numerator))]
        numerator = torch.stack(numerator)
        numerator -= self.margin
        numerator *= self.scaler
        numerator = torch.Tensor.exp(numerator)

        temp = torch.Tensor.matmul(self.W.view(self.n_speakers, self.speaker_embed_dim),speaker_embedding.transpose(0, 1)).squeeze()
        temp = self.scaler * temp
        temp = torch.Tensor.exp(temp).transpose(0, 1)
        denominator = temp.sum(1) # sum along speaker axis
        # denominator -= self.W.matmul(speaker_embedding)[true_label_index].squeeze()
        denominator = [denominator[i] - self.W.matmul(speaker_embedding[i])[true_label_index[i]] for i in range(len(denominator))]
        denominator = torch.stack(denominator)
        denominator += numerator
        denominator += self.eps
        loss = torch.Tensor.log(numerator/denominator)

        # return scalar value: averaged by batch size
        return -1*loss.mean()

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss

class Tacotron2Loss_GRL(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss_GRL, self).__init__()
        self.hparams = hparams

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, speaker_pred = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                   nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        # speaker_loss = AngularMarginSoftmaxLoss(self.hparams.classifier_hidden_dim,
        #                                         self.hparams.n_speakers,
        #                                         self.hparams.ams_margin,
        #                                         self.hparams.ams_scaler)(speaker_pred, targets[2])
        speaker_pred = speaker_pred.squeeze()
        speaker_loss = nn.CrossEntropyLoss()(speaker_pred, targets[2].cuda())
        return mel_loss + gate_loss + 0.02 * speaker_loss