class EarlyStopper:
    def __init__(
            self,
            monitor: str = "val_psnr",
            mode: str = "max",
            patience: int = 10,
            min_delta: float = 1e-3,
            verbose: bool = True
        ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

        if self.mode not in ["max", "min"]:
            raise ValueError(f"mode must be 'max' or 'min', which is {self.mode}")
        
        def __call__(self, current_score: float, epoch: int, logger) -> bool:
            if self.best_score is None:
                self.best_score = current_score
                self.best_epoch = epoch
                return False
            
            if self.mode == "max":
                is_improved = current_score > (self.best_score + self.min_delta)
            else:
                is_improved = current_score < (self.best_score - self.min_delta)
            
            if is_improved:
                self.best_score = current_score
                self.best_epoch = epoch
                self.counter = 0
                if self.verbose:
                    logger.info(f"update {self.monitor}:{current_score:.4f}(epoch {epoch})")
            else:
                self.counter += 1
                if self.verbose:
                    logger.info(f"No improvement in {self.monitor}, current count: {self.counter}/{self.patience} (Best: {self.best_score:.4f} @ epoch {self.best_epoch})")
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        logger.info(f"Early stopping triggered! Stopping training at epoch {epoch}, best {self.monitor} is {self.best_score:.4f} (epoch {self.best_epoch})")
                            
            return self.early_stop
        
