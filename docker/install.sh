#sudo docker build --build-arg LCARD_USER -t pg12 .
sudo docker build --build-arg LCARD_USER=${LCARD_USER} -t pg12 .
sudo docker run --name card-db -p 5400:5432 -d pg12

docker restart card-db
docker exec -it card-db /imdb_setup.sh

