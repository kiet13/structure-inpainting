import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import StructModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images
from .utils import write_logs_2tensorboard, write_images_2tensorboard
from .metrics import PSNR, SSIM
from torch.utils.tensorboard import SummaryWriter


class StructInpaint():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'struct'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'struct_inpaint'

        self.debug = False
        self.model_name = model_name
        self.struct_model = StructModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.ssim = SSIM()


        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_STRUCT_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_STRUCT_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_STRUCT_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.struct_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.struct_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.struct_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()

        else:
            self.struct_model.save()
            self.inpaint_model.save()


    def train(self):
        writer = SummaryWriter(self.config.TENSORBOARD_FOLDER)

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=2,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                images, structs, masks = self.cuda(*items)
                
                self.struct_model.train()
                self.inpaint_model.train()

                # struct model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.struct_model.process(images, structs, masks)
                    outputs_merged = (outputs * masks) + (structs * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(structs), self.postprocess(outputs_merged))
                    ssim = self.ssim(self.postprocess(structs), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(structs - outputs_merged)) / torch.sum(structs)).float()
                    
                    logs.append(('psnr', psnr.item()))
                    logs.append(('ssim', ssim.item()))
                    logs.append(('mae', mae.item()))
                    
                    iteration = self.struct_model.iteration


                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, structs, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(structs), self.postprocess(outputs_merged))
                    ssim = self.ssim(self.postprocess(structs), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(structs - outputs_merged)) / torch.sum(structs)).float()
                    
                    logs.append(('psnr', psnr.item()))
                    logs.append(('ssim', ssim.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # inpaint with struct model
                else:

                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.struct_model(images, structs, masks)
                        outputs = outputs * masks + structs * (1 - masks)
                    else:
                        outputs = structs

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(structs), self.postprocess(outputs_merged))
                    ssim = self.ssim(self.postprocess(structs), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(structs - outputs_merged)) / torch.sum(structs)).float()
                    
                    logs.append(('psnr', psnr.item()))
                    logs.append(('ssim', ssim.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                 # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    
                    images = self.sample()
                    write_logs_2tensorboard(writer, logs, iteration)
                    write_images_2tensorboard(writer, images, iteration)

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')
        writer.close()

    
    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.struct_model.eval()
        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, structs, masks = self.cuda(*items)

            # struct model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.struct_model.process(images, structs, masks)

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, structs, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # inpaint with struct model
            elif model == 3:
                # eval
                outputs = self.struct_model(images, structs, masks)
                outputs = outputs * masks + structs * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)



    def test(self):
        self.struct_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, structs, masks = self.cuda(*items)
            index += 1

            # struct model
            if model == 1:
                outputs = self.struct_model(images, structs, masks)
                outputs_merged = (outputs * masks) + (structs * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, structs, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with struct model / joint model
            else:
                structs = self.struct_model(images, structs, masks).detach()
                outputs = self.inpaint_model(images, structs, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)

            imsave(output, path)

            if self.debug:
                structs = self.postprocess(structs)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(structs, os.path.join(self.results_path, fname + '_struct.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.struct_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, structs, masks = self.cuda(*items)

        # struct model
        if model == 1:
            iteration = self.struct_model.iteration
            inputs = (structs * (1 - masks)) + masks
            outputs = self.struct_model(images, structs, masks)
            outputs_merged = (outputs * masks) + (structs * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, structs, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with struct model / joint model
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.struct_model(images, structs, masks).detach()
            structs = (outputs * masks + structs * (1 - masks)).detach()
            outputs = self.inpaint_model(images, structs, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
        
        grid_images = torch.cat([images, inputs, structs, outputs, outputs_merged], dim=-1)
   
        # path = os.path.join(self.samples_path, self.model_name)
        # name = os.path.join(path, str(iteration).zfill(5) + ".png")
        # create_dir(path)
        # print('\nsaving sample ' + name)
        # images.save(name)

        return grid_images

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


