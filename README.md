# CUDA_SDF_PrimalTree

논문 : https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10593351

첫 CUDA 적용 프로젝트라 부족한 점이 많음
 - KD Tree 빌드할 때 CPU에서 빌드한 후 GPU로 카피
 - Primal Tree 밀드할 때 Sample 거리 계산할 때만 GPU 사용

추후 여러 프로젝트를 진행하면서 위의 문제점 해결

참고문헌
 - J. Andreas Bærentzen and Henrik Aanæs. Generating Signed Distance Fields From Triangle Meshes. IMM-TECHNICAL REPORT-2002-21.
 - Sarah F. Frisken, Ronald N. Perry, Alyn P. Rockwood, Thouis R. Jones. Adaptively Sampled Distance Fields: A General Representation of Shape for Computer Graphics. Proceedings of the 27th annual conference on Computer graphics and interactive techniques. 2000.
 - Sylvain Lefebvre and Hugues Hoppe. Compressed Random-Access Trees for Spatially Coherent Data. The Eurographics Association 2007.
