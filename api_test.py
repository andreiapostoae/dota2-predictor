from tools.metadata import get_last_patch, get_hero_dict, get_patch
from tools.miner import mine_data

def main():
    print get_patch('7.06e')
    print get_hero_dict()
    print get_last_patch()
    mine_data(stop_at=1500)

if __name__ == '__main__':
    main()