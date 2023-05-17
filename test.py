from model import *

def main():
    config = json.load(open('./hate_model_config.json'))
    model = HateClassifier(config)

    model.load_state_dict(torch.load(f = '/hate_clf.pth'))
    dm = HateDataModule()
    dm.setup('test')

    trainer = pl.Trainer()
    trainer.test(model, dm)
    
if __name__ == '__main__':
    main()